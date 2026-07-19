#!/usr/bin/env python3
"""runpod-ollama — Single GPU inference via Ollama.

RTX 4090 / RTX 3090 (24GB VRAM) 1 台で Gemma 4 27B IT (Q4_K_M ~15GB) をロードし、
MoE 4B active params で高品質推論。workersMax=1 (単一 worker)。

VRAM Budget (RTX A5000 23GB):
  Model weights (Q4_K_M):  ~15.2 GB  (27B total, MoE)
  KV cache (q8_0, 12 slots): ~4-5 GB
  Flash Attention + runtime:  ~1 GB
  Total:                     ~21-22 GB (~90%)

Ollama は内部で llama.cpp を使用。NUM_PARALLEL スロットで
同一モデルに対する複数リクエストを GPU 内並列デコード。

Architecture:
  RunPod Serverless → handler.py (async generator)
    └─ ollama serve (background, localhost:11434)
        ├─ GGUF Q4_K_M model (~15GB, pulled on cold start)
        ├─ NUM_PARALLEL=auto (VRAM-detected: 12 for A5000, 16 for 4090/L4)
        └─ /api/chat  (stream=true → token-level SSE, stream=false → single response)

Streaming (stream=true):
  handler yields each Ollama token as an OpenAI chat.completion.chunk.
  CF Worker polls RunPod /stream/{job_id} and re-emits as SSE to client.
"""

import os
import base64
import json
import subprocess
import threading
import time
from typing import Any

import httpx
import runpod

# ── Configuration ──

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma4:26b-a4b-it-q4_K_M")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
AUTO_PARALLEL_BY_VRAM_GB = (
    (23, 16),  # 23GB+: A5000, L4, 3090, 4090  (23 covers RTX A5000's ~23GB)
    (20, 12),  # 20GB: A4500, RTX 4000 Ada
    (16, 8),   # 16GB: A4000, RTX 2000 Ada
)
MIN_PARALLEL = 4
MAX_VISION_IMAGE_BYTES = int(os.environ.get("MAX_VISION_IMAGE_BYTES", str(12 * 1024 * 1024)))
IMAGE_FETCH_TIMEOUT_SECONDS = float(os.environ.get("VISION_IMAGE_FETCH_TIMEOUT_SECONDS", "60"))

# ── Ollama Server Management ──

_ollama_proc: subprocess.Popen | None = None
_ollama_ready = False
_runtime_parallel: int | None = None
_runtime_concurrency: int | None = None
_gpu_name = "unknown"
_gpu_vram_gb = 0
_startup_lock = threading.Lock()


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def _detect_gpu_vram_gb() -> int:
    """Return first GPU VRAM in GB (rounded down)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        return max(0, int(out.splitlines()[0].strip()) // 1024) if out else 0
    except Exception:
        return 0


def _auto_parallel_from_vram(vram_gb: int) -> int:
    for min_vram, parallel in AUTO_PARALLEL_BY_VRAM_GB:
        if vram_gb >= min_vram:
            return parallel
    return MIN_PARALLEL


def _resolve_parallel(vram_gb: int) -> int:
    raw = os.environ.get("OLLAMA_NUM_PARALLEL", "auto").strip().lower()
    return _auto_parallel_from_vram(vram_gb) if raw == "auto" else int(raw)


def _resolve_concurrency(parallel: int) -> int:
    raw = os.environ.get("CONCURRENCY", "auto").strip().lower()
    return parallel if raw == "auto" else int(raw)


def ensure_ollama():
    """Start ollama serve as background process (once per worker lifetime)."""
    global _ollama_proc, _ollama_ready, _runtime_parallel, _runtime_concurrency, _gpu_name, _gpu_vram_gb
    with _startup_lock:
        if _ollama_ready:
            return

        _gpu_name = _detect_gpu_name()
        _gpu_vram_gb = _detect_gpu_vram_gb()

        # VRAM guard: fail fast with clear message instead of slow OOM/warmup failure
        print(f"[runpod-ollama] detected gpu={_gpu_name} vram={_gpu_vram_gb}GB model={MODEL_NAME}")
        # Only guard when VRAM detection succeeded (>0) and model is large
        # gemma4:26b-a4b needs ~15.2GB (Q4_K_M) + KV cache — require ≥18GB
        if 0 < _gpu_vram_gb < 18 and "26b" in MODEL_NAME.lower():
            raise RuntimeError(
                f"VRAM_INSUFFICIENT: gpu={_gpu_name} vram={_gpu_vram_gb}GB < 18GB required for {MODEL_NAME}"
            )

        _runtime_parallel = _resolve_parallel(_gpu_vram_gb)
        _runtime_concurrency = _resolve_concurrency(_runtime_parallel)

        env = {
            **os.environ,
            "OLLAMA_NUM_PARALLEL": str(_runtime_parallel),
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
        }

        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        client = httpx.Client(timeout=120)
        for _ in range(60):
            try:
                if client.get(f"{OLLAMA_HOST}/api/version").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("ollama serve failed to start within 120s")

        print(
            f"[runpod-ollama] gpu={_gpu_name} vram={_gpu_vram_gb}GB "
            f"parallel={_runtime_parallel} concurrency={_runtime_concurrency}"
        )
        print(f"[runpod-ollama] ensuring model exists: {MODEL_NAME}")
        client.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": MODEL_NAME, "stream": False},
            timeout=900,
        ).raise_for_status()

        print(f"[runpod-ollama] loading {MODEL_NAME} into GPU...")
        resp = client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
            timeout=300,
        )
        if resp.status_code == 404:
            print(f"[runpod-ollama] warmup 404; retrying pull for {MODEL_NAME}")
            client.post(
                f"{OLLAMA_HOST}/api/pull",
                json={"model": MODEL_NAME, "stream": False},
                timeout=900,
            ).raise_for_status()
            resp = client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
                timeout=300,
            )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = json.dumps(resp.json(), ensure_ascii=False)
            except Exception:
                pass
            raise RuntimeError(
                f"ollama warmup failed status={resp.status_code} model={MODEL_NAME} detail={detail}"
            )
        print(f"[runpod-ollama] model loaded, {_runtime_parallel} parallel slots active")

        _ollama_ready = True
        client.close()


# ── RunPod Handler (async generator) ──


def _data_url_base64(url: str) -> str | None:
    prefix, sep, payload = url.partition(",")
    if not sep or ";base64" not in prefix.lower():
        return None
    return payload.strip()


async def _fetch_image_base64(client: httpx.AsyncClient, url: str) -> str:
    data_url = _data_url_base64(url)
    if data_url is not None:
        return data_url

    if not url.startswith(("https://", "http://")):
        raise ValueError("image_url must be an http(s) URL or base64 data URL")

    async with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > MAX_VISION_IMAGE_BYTES:
                raise ValueError(f"image too large: {total} bytes > {MAX_VISION_IMAGE_BYTES}")
            chunks.append(chunk)
    return base64.b64encode(b"".join(chunks)).decode("ascii")


def _image_url_from_part(part: dict[str, Any]) -> str:
    value = part.get("image_url")
    if isinstance(value, dict):
        url = value.get("url")
        return url if isinstance(url, str) else ""
    if isinstance(value, str):
        return value

    camel_value = part.get("imageUrl")
    if isinstance(camel_value, dict):
        url = camel_value.get("url")
        return url if isinstance(url, str) else ""
    if isinstance(camel_value, str):
        return camel_value
    return ""


async def _normalize_messages_for_ollama(messages: Any) -> tuple[list[dict[str, Any]], int]:
    """Convert OpenAI multimodal chat content parts to Ollama chat messages."""
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")

    normalized: list[dict[str, Any]] = []
    image_count = 0
    async with httpx.AsyncClient(timeout=IMAGE_FETCH_TIMEOUT_SECONDS) as client:
        for message in messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "user")
            content = message.get("content", "")
            out: dict[str, Any] = {"role": role}

            if isinstance(content, str):
                out["content"] = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                images: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                        continue
                    if not isinstance(part, dict):
                        continue

                    part_type = str(part.get("type") or "").lower()
                    if part_type == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                    elif part_type in {"image_url", "imageurl", "input_image"}:
                        url = _image_url_from_part(part)
                        if url:
                            images.append(await _fetch_image_base64(client, url))

                out["content"] = "\n".join(t for t in text_parts if t)
                if images:
                    out["images"] = images
                    image_count += len(images)
            else:
                out["content"] = str(content or "")

            normalized.append(out)

    return normalized, image_count


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless handler — proxy to local Ollama.

    Returns an OpenAI-compatible chat completion response.

    stream=false (default):
      Returns one complete OpenAI chat.completion response.

    stream=true:
      Uses Ollama streaming internally and returns one complete response after
      accumulating chunks. RunPod does not reliably surface async generator
      yields in /status output, so the gateway depends on return values.

    Input:
    {
        "input": {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": false,
            "think": false
        }
    }
    """
    try:
        ensure_ollama()
    except Exception as e:
        return {
            "error": str(e),
            "error_stage": "startup",
            "model": MODEL_NAME,
        }

    job_input = job.get("input", {})
    messages = job_input.get("messages", [])
    if not messages:
        return {"error": "messages is required"}

    max_tokens = int(job_input.get("max_tokens", 2048))
    temperature = float(job_input.get("temperature", 0.7))
    top_p = float(job_input.get("top_p", 0.95))
    think = bool(job_input.get("think", False))
    stream_mode = bool(job_input.get("stream", False))
    try:
        messages, vision_image_count = await _normalize_messages_for_ollama(messages)
    except Exception as e:
        return {
            "error": str(e),
            "error_stage": "vision_input_normalize",
            "model": MODEL_NAME,
        }

    metadata = {
        "x_engine": "ollama",
        "x_gpu": _gpu_name,
        "x_gpu_vram_gb": _gpu_vram_gb,
        "x_parallel_slots": _runtime_parallel,
        "x_concurrency": _runtime_concurrency,
        "x_quantization": "Q4_K_M",
        "x_vision_images": vision_image_count,
    }

    ollama_params = {
        "model": MODEL_NAME,
        "messages": messages,
        "think": think,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    start_time = time.monotonic()
    cmpl_id = f"chatcmpl-runpod-{int(start_time * 1000)}"

    if stream_mode:
        # ── Streaming internally: accumulate one final OpenAI response ──
        content_parts: list[str] = []
        last_data: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json={**ollama_params, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")
                    return {"error": f"ollama error {resp.status_code}", "detail": detail}

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    if content:
                        content_parts.append(content)

                    if done:
                        last_data = data
                        break

        gpu_time_ms = int((time.monotonic() - start_time) * 1000)
        content = "".join(content_parts)
        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": last_data.get("prompt_eval_count", 0),
                "completion_tokens": last_data.get("eval_count", 0),
                "total_tokens": last_data.get("prompt_eval_count", 0) + last_data.get("eval_count", 0),
            },
            "x_gpu_time_ms": gpu_time_ms,
            **metadata,
        }

    else:
        # ── Non-streaming: yield single complete response ──
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={**ollama_params, "stream": False},
        )

        if resp.status_code != 200:
            return {"error": f"ollama error {resp.status_code}", "detail": resp.text}

        ollama_resp = resp.json()
        gpu_time_ms = int((time.monotonic() - start_time) * 1000)
        content = ollama_resp.get("message", {}).get("content", "")

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                "completion_tokens": ollama_resp.get("eval_count", 0),
                "total_tokens": ollama_resp.get("prompt_eval_count", 0) + ollama_resp.get("eval_count", 0),
            },
            "x_gpu_time_ms": gpu_time_ms,
            **metadata,
        }


# ── RunPod Entry ──

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda current: _resolve_concurrency(
        _resolve_parallel(_detect_gpu_vram_gb())
    ),
})
