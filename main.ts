// Deterministic free AI model aggregator server and CLI entrypoint (`node main.ts [--list]`).
// Implements OpenAI (/v1/chat/completions, /v1/models) and Anthropic (/v1/messages, /v1/models)
// endpoints backed by probed upstream OpenAI, Anthropic, and Gemini providers with scarcity-aware
// quality routing and automatic failover.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
} from './lib/providers.ts';
import {
  createRouterState,
  recordProviderOutcome,
  selectNextCandidate,
  type CandidateModel,
  type RouterState,
} from './lib/router.ts';
import {
  anthropicModelList,
  buildUpstreamRequest,
  detectRequestNeeds,
  openAIModelList,
  parseClientRequest,
  readUpstreamResponse,
  renderClientJson,
  renderClientSse,
  type ClientWire,
} from './lib/wire.ts';

const UPSTREAM_TIMEOUT_MS = 15_000;
const WORK_POLL_INTERVAL_MS = 100;

async function collectBriefAndProbes(briefPath: string, envUrl: string): Promise<{
  brief: Brief;
  probes: Probe[];
  catalog: CatalogReport;
}> {
  const brief = readBrief(briefPath);
  if (brief.error !== null) {
    return { brief, probes: [], catalog: { providers: [] } };
  }
  const probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(envUrl, provider)),
  );
  const catalog = buildCatalog(envUrl, probes, brief.ratings);
  return { brief, probes, catalog };
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body, 'utf8'),
  });
  res.end(body);
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk, 'utf8') : chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (raw.trim() === '') return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function callUpstreamCandidate(
  envUrl: string,
  cand: CandidateModel,
  path: string,
  body: Json,
  stream: boolean,
): Promise<{ ok: boolean; status: number; res?: Response; error?: string }> {
  const base = joinBase(envUrl, cand.provider.base);
  const url = `${base}${path}`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(cand.provider.auth, cand.provider.key),
  };
  if (cand.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
  }
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    return { ok: true, status: res.status, res };
  } catch (err) {
    return { ok: false, status: 502, error: (err as Error).message };
  }
}

async function handleInference(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: ClientWire,
  envUrl: string,
  routerState: RouterState,
): Promise<void> {
  const rawBody = await readJsonBody(req);
  if (rawBody === null) {
    sendJson(res, 400, { error: { message: 'Invalid JSON request body', type: 'invalid_request_error' } });
    return;
  }

  const canonicalReq = parseClientRequest(clientWire, rawBody);
  const detected = detectRequestNeeds(canonicalReq);
  const needs = {
    clientWire,
    stream: canonicalReq.stream,
    tools: detected.tools,
    vision: detected.vision,
    json: detected.json,
    strictJsonWire: detected.strictJsonWire,
    minContext: detected.minContext,
  };

  const tried = new Set<string>();
  const retriedOnce = new Set<string>();
  const maxAttempts = Math.max(4, routerState.providers.length + 2);
  let lastError = 'No eligible free provider succeeded';

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let cand = selectNextCandidate(routerState, needs, tried);

    // If every distinct provider has failed once and at least one returned a transient error (e.g. 429),
    // allow a single deterministic fallback pass before giving up on a scarce capability request.
    if (cand === null) {
      const transientExcluded = new Set<string>();
      for (const ps of routerState.providers) {
        if (tried.has(ps.entry.id)) {
          if (retriedOnce.has(ps.entry.id) || ps.lastErrorStatus === 402 || ps.consecFailures >= 2) {
            transientExcluded.add(ps.entry.id);
          }
        }
      }
      cand = selectNextCandidate(routerState, needs, transientExcluded);
      if (cand !== null) retriedOnce.add(cand.provider.id);
    }

    if (cand === null) break;
    tried.add(cand.provider.id);

    const { path, body } = buildUpstreamRequest(canonicalReq, cand.provider.wire, cand.modelId);
    const transport = await callUpstreamCandidate(envUrl, cand, path, body, canonicalReq.stream);

    if (!transport.ok || transport.res === undefined) {
      lastError = transport.error ?? 'Upstream connection error';
      recordProviderOutcome(routerState, cand.provider.id, false, transport.status, false);
      continue;
    }

    const parsed = await readUpstreamResponse(cand.provider.wire, canonicalReq.stream, transport.res);
    if (!parsed.ok || parsed.response === undefined) {
      lastError = parsed.error ?? `HTTP ${parsed.status}`;
      recordProviderOutcome(routerState, cand.provider.id, false, parsed.status, false);
      continue;
    }

    recordProviderOutcome(routerState, cand.provider.id, true, parsed.status, true);

    if (canonicalReq.stream) {
      const ssePayload = renderClientSse(clientWire, canonicalReq.clientModel, parsed.response);
      res.writeHead(200, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      res.end(ssePayload);
    } else {
      const jsonPayload = renderClientJson(clientWire, canonicalReq.clientModel, parsed.response);
      sendJson(res, 200, jsonPayload);
    }
    return;
  }

  // Advance manifest cursor even if all upstreams failed so subsequent requests stay aligned.
  recordProviderOutcome(routerState, '__none__', false, 502, true);
  if (clientWire === 'anthropic') {
    sendJson(res, 502, { type: 'error', error: { type: 'api_error', message: lastError } });
  } else {
    sendJson(res, 502, { error: { message: lastError, type: 'upstream_error' } });
  }
}

async function run(): Promise<void> {
  const listOnly = process.argv.includes('--list');
  const briefPath = (process.env.MAKEFUN_PROMPT_FILE ?? '').trim();
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').trim();

  if (listOnly) {
    if (briefPath === '' || envUrl === '') {
      process.stdout.write('{"providers":[]}\n');
      return;
    }
    const { catalog } = await collectBriefAndProbes(briefPath, envUrl);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  if (briefPath === '' || envUrl === '') {
    process.stderr.write('MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n');
    process.exit(1);
  }

  const { brief, probes, catalog } = await collectBriefAndProbes(briefPath, envUrl);
  if (brief.error !== null) {
    process.stderr.write(`${brief.error}\n`);
    process.exit(1);
  }

  const routerState = createRouterState(brief, probes);

  // Serialize requests through a promise chain to preserve deterministic sequential state updates.
  let queue: Promise<void> = Promise.resolve();

  const server: Server = createServer((req, res) => {
    const pathname = new URL(req.url ?? '/', 'http://127.0.0.1').pathname;
    const method = (req.method ?? 'GET').toUpperCase();

    if (method === 'GET' && pathname === '/v1/models') {
      const isAnthropic = req.headers['x-api-key'] !== undefined;
      sendJson(res, 200, isAnthropic ? anthropicModelList(catalog) : openAIModelList(catalog));
      return;
    }

    if (method === 'GET' && pathname === '/health') {
      sendJson(res, 200, { ok: true, providers: catalog.providers.length });
      return;
    }

    if (method === 'POST' && (pathname === '/v1/chat/completions' || pathname === '/v1/messages')) {
      const clientWire: ClientWire = pathname === '/v1/messages' ? 'anthropic' : 'openai';
      queue = queue.then(() => handleInference(req, res, clientWire, envUrl, routerState));
      return;
    }

    sendJson(res, 404, { error: { message: `Unknown route ${method} ${pathname}`, type: 'not_found_error' } });
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });

  const addr = server.address() as AddressInfo;
  const registerUrl = `${joinBase(envUrl, '')}/register`;
  const regRes = await fetch(registerUrl, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port: addr.port, catalog }),
  });
  if (!regRes.ok) {
    const msg = await regRes.text().catch(() => '');
    process.stderr.write(`POST /register failed (${regRes.status}): ${msg}\n`);
    server.close();
    process.exit(1);
  }

  const workUrl = `${joinBase(envUrl, '')}/work`;
  const timer = setInterval(async () => {
    try {
      const res = await fetch(workUrl);
      if (!res.ok) return;
      const body = (await res.json()) as { done?: boolean };
      if (body.done === true) {
        clearInterval(timer);
        await queue;
        server.close(() => process.exit(0));
      }
    } catch {
      // Env loop transient hiccup: wait for next tick.
    }
  }, WORK_POLL_INTERVAL_MS);
  timer.unref();
}

run().catch((err) => {
  process.stderr.write(`${(err as Error).stack ?? String(err)}\n`);
  process.exit(1);
});
