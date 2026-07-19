/**
 * runpod CF Worker — Stateless API Gateway.
 *
 * Proxies OpenAI-compatible requests to RunPod Serverless endpoint.
 * RunPod handles auto-scaling and load balancing across RTX 3090 workers.
 *
 * Architecture:
 *   Client → CF Worker (auth + transform) → RunPod Serverless API → Ollama (CUDA)
 *
 * Contracts:
 *   - /v1/chat/completions (OpenAI-compatible → RunPod runsync or stream)
 *   - /v1/models
 *   - /health
 *
 * Streaming (stream: true):
 *   1. POST /run (async job) → job_id
 *   2. Poll /stream/{job_id} at 100ms intervals
 *   3. Each yielded chunk from handler.py re-emitted as OpenAI SSE chunk
 *   4. Terminated with "data: [DONE]"
 *
 *   Note: cold start (~120s) delays first token. Warn clients accordingly.
 */

import { Hono } from "hono";
import { RUNPOD_DEFAULT_MODEL } from "./llm-models";

/** Model ID — synced with handler.py MODEL_NAME / OLLAMA_MODEL. */
const MODEL_ID = RUNPOD_DEFAULT_MODEL;
const ANSWER_WITH_KNOWLEDGE_NSID = "com.etzhayyim.apps.llm.answerWithKnowledge";
const PUBLIC_MODEL_ALIASES = ["gemma4-runpod", "tier0-runpod", MODEL_ID] as const;

interface Env {
  /** RunPod Serverless endpoint ID (e.g. "abc123def456"). */
  RUNPOD_ENDPOINT_ID: string;
  /** RunPod API key. */
  RUNPOD_API_KEY?: string;
  /** API key for client authentication. */
  RUNPOD_GATEWAY_API_KEY?: string;
}

const HARDCODED_GATEWAY_KEY =
  "rpgw_7kXm3Nv8QwPf2RsYtUeH4JcLbA9DzGiO6WhK1MpV5nBx";

const app = new Hono<{ Bindings: Env }>();

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, x-kotodama-verified",
  "Access-Control-Max-Age": "86400",
};

// ── Auth ──

function validateApiKey(env: Env, request: Request): boolean {
  const kotodama = request.headers.get("x-kotodama-verified");
  if (kotodama === "true") return true;

  const authHeader = request.headers.get("Authorization");
  const xApiKey = request.headers.get("x-api-key");
  const expectedKey = env.RUNPOD_GATEWAY_API_KEY || HARDCODED_GATEWAY_KEY;
  const providedKey = authHeader?.replace("Bearer ", "") || xApiKey || "";
  return providedKey === expectedKey;
}

const requireAuth: Parameters<typeof app.use>[1] = async (c, next) => {
  if (!validateApiKey(c.env, c.req.raw)) {
    return c.json({ error: "unauthorized" }, 401);
  }
  await next();
};

// ── RunPod Proxy ──

/**
 * Non-streaming: POST /run → poll /status/{job_id} → return complete response.
 *
 * Uses /run + polling instead of /runsync because /runsync has a ~90s server-side
 * timeout and returns IN_PROGRESS with empty body if the job hasn't completed yet.
 */
async function runpodInfer(
  env: Env,
  input: Record<string, unknown>,
  timeoutMs = 300_000,
): Promise<Response> {
  if (!env.RUNPOD_API_KEY) {
    return new Response(
      JSON.stringify({ error: "runpod_api_key_missing" }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  // 1. Submit async job
  let jobId: string;
  try {
    const runResp = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.RUNPOD_API_KEY}`,
        },
        body: JSON.stringify({ input }),
        signal: AbortSignal.timeout(15_000),
      },
    );
    if (!runResp.ok) {
      const errBody = await runResp.text();
      return new Response(
        JSON.stringify({ error: "runpod_error", status: runResp.status, detail: errBody }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      );
    }
    const runData = await runResp.json() as { id: string; status: string };
    jobId = runData.id;
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "runpod_unavailable", detail: String(err) }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  // 2. Poll /status/{jobId} until COMPLETED or FAILED
  const statusUrl = `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/status/${jobId}`;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    await sleep(2_000);

    let result: { id: string; status: string; output?: unknown; error?: string };
    try {
      const statusResp = await fetch(statusUrl, {
        headers: { "Authorization": `Bearer ${env.RUNPOD_API_KEY}` },
        signal: AbortSignal.timeout(15_000),
      });
      if (!statusResp.ok) {
        await sleep(3_000);
        continue;
      }
      result = await statusResp.json() as typeof result;
    } catch {
      await sleep(3_000);
      continue;
    }

    if (result.status === "COMPLETED") {
      // output from async generator = array of yielded values; take last non-error item
      let output = result.output;
      if (Array.isArray(output)) {
        output = output[output.length - 1] ?? output[0];
      }
      return new Response(JSON.stringify(output), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "x-runpod-id": result.id,
          "x-runpod-status": result.status,
        },
      });
    }

    if (result.status === "FAILED" || result.status === "CANCELLED") {
      return new Response(
        JSON.stringify({ error: "inference_failed", detail: result.error }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      );
    }

    // IN_QUEUE or IN_PROGRESS → keep polling
  }

  return new Response(
    JSON.stringify({ error: "timeout", jobId }),
    { status: 504, headers: { "Content-Type": "application/json" } },
  );
}

/**
 * Streaming: POST /run → poll /stream/{job_id} NDJSON → OpenAI SSE.
 *
 * RunPod /stream/{job_id} is a polling endpoint (not a long-lived connection):
 *   - Each call returns the current job state + any new stream chunks, then closes.
 *   - We poll repeatedly until status is COMPLETED / FAILED / CANCELLED.
 *   - Each response is one NDJSON line: { status, stream: [{ output: <chunk> }, ...] }
 *
 * Chunks yielded by handler.py are accumulated in stream[].output.
 * We emit each as an OpenAI SSE event, then emit [DONE] on terminal status.
 */
async function runpodStream(
  env: Env,
  input: Record<string, unknown>,
  timeoutMs = 300_000,
): Promise<Response> {
  if (!env.RUNPOD_API_KEY) {
    return new Response(
      JSON.stringify({ error: "runpod_api_key_missing" }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  // 1. Submit async job
  let jobId: string;
  try {
    const runResp = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.RUNPOD_API_KEY}`,
        },
        body: JSON.stringify({ input }),
        signal: AbortSignal.timeout(15_000),
      },
    );
    if (!runResp.ok) {
      const detail = await runResp.text();
      return new Response(
        JSON.stringify({ error: "stream_submit_failed", detail }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      );
    }
    const runData = await runResp.json() as { id: string; status: string };
    jobId = runData.id;
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "runpod_unavailable", detail: String(err) }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  // 2. Poll /stream/{jobId} in a ReadableStream — each poll returns new chunks.
  const encoder = new TextEncoder();
  const streamUrl = `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/stream/${jobId}`;
  const authHeader = `Bearer ${env.RUNPOD_API_KEY}`;

  // Capture jobId + env references in closure for the ReadableStream start() fn.
  const capturedJobId = jobId;
  const capturedStreamUrl = streamUrl;
  const capturedAuth = authHeader;

  const readable = new ReadableStream<Uint8Array>({
    async start(controller) {
      const deadline = Date.now() + timeoutMs;
      let emittedJobId = false;
      let emittedChunkCount = 0;
      let seenChunkIndices = new Set<number>();

      let lastDataSent = Date.now();
      const enq = (s: string) => {
        controller.enqueue(encoder.encode(s));
        lastDataSent = Date.now();
      };

      while (Date.now() < deadline) {
        // Send SSE keepalive comment if >15s since last data sent to client
        if (Date.now() - lastDataSent > 15_000) {
          enq(": keepalive\n\n");
        }

        let pollResp: Response;
        try {
          pollResp = await fetch(capturedStreamUrl, {
            headers: { "Authorization": capturedAuth },
            signal: AbortSignal.timeout(30_000),
          });
        } catch {
          await sleep(2_000);
          continue;
        }

        if (!pollResp.ok) {
          await sleep(2_000);
          continue;
        }

        const text = await pollResp.text();
        let terminal = false;

        for (const line of text.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          let event: {
            status?: string;
            stream?: Array<{ index?: number; output: unknown }>;
            output?: unknown;
            error?: string;
          };
          try { event = JSON.parse(trimmed); } catch { continue; }

          // Emit job_id metadata chunk once on first real response
          if (!emittedJobId) {
            enq(`data: ${JSON.stringify({ id: capturedJobId, object: "chat.completion.chunk", model: MODEL_ID, choices: [], x_runpod_job_id: capturedJobId })}\n\n`);
            emittedJobId = true;
          }

          // Emit new stream items (deduplicate by index if present)
          for (const item of event.stream ?? []) {
            const idx = item.index ?? emittedChunkCount;
            if (seenChunkIndices.has(idx)) continue;
            seenChunkIndices.add(idx);
            emittedChunkCount++;
            // RunPod wraps each yield as { output: <yielded_value> }
            const openaiChunk = (item.output as Record<string, unknown>) ?? item;
            enq(`data: ${JSON.stringify(openaiChunk)}\n\n`);
          }

          const status = event.status ?? "";
          terminal = status === "COMPLETED" || status === "FAILED" || status === "CANCELLED";

          if (terminal) {
            // Fallback: sync handler (no stream items) → emit full output as one chunk
            if ((event.stream ?? []).length === 0 && event.output) {
              enq(`data: ${JSON.stringify(event.output)}\n\n`);
            }
            // Forward RunPod error detail on FAILED (e.g. VRAM_INSUFFICIENT, warmup error)
            if (status === "FAILED" && event.error) {
              enq(`data: ${JSON.stringify({ error: "runpod_job_failed", detail: event.error, model: MODEL_ID })}\n\n`);
            }
          }
        }

        if (terminal) break;

        // RunPod polling interval: 2s during warmup, 500ms while running
        await sleep(emittedChunkCount > 0 ? 500 : 2_000);
      }

      enq("data: [DONE]\n\n");
      controller.close();
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
      "x-runpod-id": capturedJobId,
    },
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function proxyAnswerWithKnowledge(env: Env, request: Request): Promise<Response> {
  void env;
  void request;
  return new Response(
    JSON.stringify({
      error: "unsupported_route",
      message: `${ANSWER_WITH_KNOWLEDGE_NSID} is not served by the independent RunPod llm.etzhayyim.com gateway.`,
    }),
    {
      status: 404,
      headers: { "Content-Type": "application/json", ...CORS_HEADERS },
    },
  );
}

// ── Debug Endpoints (temporary) ──

/** Submit a test job and return jobId immediately (well under CF 30s wall-clock limit). */
app.post("/_debug/submit", requireAuth, async (c) => {
  const body = await c.req.json() as Record<string, unknown>;
  const input = {
    messages: body.messages ?? [{ role: "user", content: "hi" }],
    max_tokens: 20,
    temperature: 0.7,
    top_p: 0.95,
    stop: [],
    stream: true,
    think: false,
  };

  const runResp = await fetch(`https://api.runpod.ai/v2/${c.env.RUNPOD_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}` },
    body: JSON.stringify({ input }),
    signal: AbortSignal.timeout(15_000),
  });
  const runData = await runResp.json();
  return c.json({ runData, hint: "Poll /_debug/status/:jobId to check progress" });
});

/** Poll job status + stream once — call every few seconds to see progress and errors. */
app.get("/_debug/status/:jobId", requireAuth, async (c) => {
  const jobId = c.req.param("jobId");
  const [statusResp, streamResp] = await Promise.all([
    fetch(`https://api.runpod.ai/v2/${c.env.RUNPOD_ENDPOINT_ID}/status/${jobId}`, {
      headers: { "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}` },
      signal: AbortSignal.timeout(10_000),
    }),
    fetch(`https://api.runpod.ai/v2/${c.env.RUNPOD_ENDPOINT_ID}/stream/${jobId}`, {
      headers: { "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}` },
      signal: AbortSignal.timeout(10_000),
    }),
  ]);
  const statusData = await statusResp.json();
  const streamText = await streamResp.text();
  return c.json({ jobId, status: statusData, streamStatus: streamResp.status, streamText });
});

/**
 * Update RunPod template image via GraphQL — uses stored RUNPOD_API_KEY secret.
 * Body: { templateId: string, imageName: string, envVars?: { key: string; value: string }[] }
 */
app.post("/_debug/update-template", requireAuth, async (c) => {
  if (!c.env.RUNPOD_API_KEY) return c.json({ error: "runpod_api_key_missing" }, 503);
  const body = await c.req.json() as {
    templateId: string;
    name: string;
    imageName: string;
    dockerArgs: string;
    containerDiskInGb: number;
    volumeInGb: number;
    envVars: { key: string; value: string }[];
  };

  const query = `
    mutation saveTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id name imageName dockerArgs containerDiskInGb volumeInGb env { key value }
      }
    }
  `;

  const resp = await fetch("https://api.runpod.io/graphql", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({
      query,
      variables: {
        input: {
          id: body.templateId,
          name: body.name,
          imageName: body.imageName,
          dockerArgs: body.dockerArgs,
          containerDiskInGb: body.containerDiskInGb,
          volumeInGb: body.volumeInGb,
          env: body.envVars,
        },
      },
    }),
    signal: AbortSignal.timeout(15_000),
  });

  const data = await resp.json();
  return c.json({ httpStatus: resp.status, data });
});

/** Fetch RunPod template details via GraphQL — POST with { templateId }. */
app.post("/_debug/get-template", requireAuth, async (c) => {
  if (!c.env.RUNPOD_API_KEY) return c.json({ error: "runpod_api_key_missing" }, 503);
  const body = await c.req.json() as { templateId: string };
  const query = `
    query {
      myself {
        podTemplates {
          id name imageName dockerArgs containerDiskInGb volumeInGb
          env { key value }
        }
      }
    }
  `;
  const resp = await fetch("https://api.runpod.io/graphql", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ query }),
    signal: AbortSignal.timeout(10_000),
  });
  const data = await resp.json() as { data?: { myself?: { podTemplates?: unknown[] } } };
  const templates = data?.data?.myself?.podTemplates ?? [];
  const target = (templates as Array<{ id: string }>).find(t => t.id === body.templateId);
  return c.json({ templateId: body.templateId, found: target ?? null });
});

// ── Public Endpoints ──

app.get("/health", async (c) => {
  if (!c.env.RUNPOD_API_KEY) {
    return c.json({
      status: "degraded",
      backend: "runpod-ollama",
      model: MODEL_ID,
      error: "runpod_api_key_missing",
    }, 503);
  }
  try {
    const url = `https://api.runpod.ai/v2/${c.env.RUNPOD_ENDPOINT_ID}/health`;
    const resp = await fetch(url, {
      headers: { "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}` },
    });
    const data = await resp.json() as Record<string, unknown>;
    return c.json({
      status: "ok",
      backend: "runpod-ollama",
      model: MODEL_ID,
      streaming: true,
      runpod: data,
    });
  } catch (err) {
    return c.json({ status: "degraded", backend: "runpod-ollama", error: String(err) });
  }
});

app.get("/_app/meta", (c) =>
  c.json({
    name: "runpod",
    nanoid: "r9np0d01",
    version: "1.1.0",
    backend: "runpod-ollama",
    model: MODEL_ID,
    capabilities: ["llm-inference", "cuda", "continuous-batching", "sse-streaming"],
  }),
);

app.options(`/xrpc/${ANSWER_WITH_KNOWLEDGE_NSID}`, (c) => new Response(null, { status: 204, headers: CORS_HEADERS }));

app.post(`/xrpc/${ANSWER_WITH_KNOWLEDGE_NSID}`, async (c) => proxyAnswerWithKnowledge(c.env, c.req.raw));

// ── OpenAI-compatible API ──

app.get("/v1/models", requireAuth, (c) =>
  c.json({
    object: "list",
    data: PUBLIC_MODEL_ALIASES.map((id) => ({
      id,
      object: "model",
      owned_by: "google",
      backend: "runpod-ollama",
      root: MODEL_ID,
      max_tokens: 8192,
      context_window: 8192,
      streaming: true,
    })),
  }),
);

app.post("/v1/chat/completions", requireAuth, async (c) => {
  const body = await c.req.json() as Record<string, unknown>;
  const wantsStream = body.stream === true;

  const input = {
    messages: body.messages,
    max_tokens: body.max_tokens ?? 2048,
    temperature: body.temperature ?? 0.7,
    top_p: body.top_p ?? 0.95,
    stop: body.stop ?? [],
    stream: wantsStream,
    think: body.think ?? false,
  };

  return wantsStream
    ? runpodStream(c.env, input)
    : runpodInfer(c.env, input);
});

// Long-path aliases
app.get("/api/openai/v1/models", requireAuth, (c) =>
  c.json({
    object: "list",
    data: PUBLIC_MODEL_ALIASES.map((id) => ({ id, object: "model", owned_by: "google", root: MODEL_ID })),
  }),
);

app.post("/api/openai/v1/chat/completions", requireAuth, async (c) => {
  const body = await c.req.json() as Record<string, unknown>;
  const wantsStream = body.stream === true;

  const input = {
    messages: body.messages,
    max_tokens: body.max_tokens ?? 2048,
    temperature: body.temperature ?? 0.7,
    top_p: body.top_p ?? 0.95,
    stop: body.stop ?? [],
    stream: wantsStream,
    think: body.think ?? false,
  };

  return wantsStream
    ? runpodStream(c.env, input)
    : runpodInfer(c.env, input);
});

// ── XRPC Compatibility ──

app.post("/xrpc/etzhayyim.runpod.v1.RunpodQueryService/GetClusterStatus", requireAuth, async (c) => {
  try {
    const url = `https://api.runpod.ai/v2/${c.env.RUNPOD_ENDPOINT_ID}/health`;
    const resp = await fetch(url, {
      headers: { "Authorization": `Bearer ${c.env.RUNPOD_API_KEY}` },
    });
    const data = await resp.json() as {
      workers?: { idle?: number; running?: number; throttled?: number };
      jobs?: { completed?: number; failed?: number; inQueue?: number; inProgress?: number };
    };
    return c.json({
      totalWorkers: (data.workers?.idle ?? 0) + (data.workers?.running ?? 0),
      idleWorkers: data.workers?.idle ?? 0,
      runningWorkers: data.workers?.running ?? 0,
      throttledWorkers: data.workers?.throttled ?? 0,
      completedJobs: data.jobs?.completed ?? 0,
      failedJobs: data.jobs?.failed ?? 0,
      inQueue: data.jobs?.inQueue ?? 0,
      inProgress: data.jobs?.inProgress ?? 0,
      model: MODEL_ID,
      engine: "ollama",
      streaming: true,
    });
  } catch (err) {
    return c.json({ error: String(err) }, 503);
  }
});

// ── Fallback ──

app.all("*", (c) => c.json({ error: "not_found", path: c.req.path }, 404));

export default app;
