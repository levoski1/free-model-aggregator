import assert from 'node:assert/strict';
import test from 'node:test';
import {
  anthropicModelList,
  buildUpstreamRequest,
  openAIModelList,
  parseClientRequest,
  renderClientJson,
  renderClientSse,
} from '../lib/wire.ts';
import { createRouterState, recordProviderOutcome, selectNextCandidate } from '../lib/router.ts';
import type { Brief, CatalogReport, Probe, ProviderDirectoryEntry } from '../lib/providers.ts';

function makeProvider(id: string, wire: 'openai' | 'anthropic' | 'gemini'): ProviderDirectoryEntry {
  return {
    id,
    name: id,
    wire,
    auth: wire === 'anthropic' ? 'x-api-key' : wire === 'gemini' ? 'x-goog-api-key' : 'bearer',
    base: `/p/${id}`,
    key_env: `${id.toUpperCase()}_KEY`,
    key: 'test-key',
  };
}

test('OpenAI request translates faithfully to Gemini with images, tools, max_tokens, stop, json_mode', () => {
  const req = parseClientRequest('openai', {
    model: 'auto',
    messages: [
      { role: 'system', content: 'Be terse.' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Inspect this chart' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
        ],
      },
    ],
    max_tokens: 200,
    stop: ['###'],
    temperature: 0.4,
    response_format: { type: 'json_object' },
    tools: [
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Lookup item',
          parameters: { type: 'object', properties: { id: { type: 'string' } } },
        },
      },
    ],
    stream: true,
  });

  const { path, body } = buildUpstreamRequest(req, 'gemini', 'gemini-3.8-flash');
  assert.equal(path, '/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse');
  assert.deepEqual(body.systemInstruction, { parts: [{ text: 'Be terse.' }] });
  const genCfg = body.generationConfig as Record<string, unknown>;
  assert.equal(genCfg.maxOutputTokens, 200);
  assert.deepEqual(genCfg.stopSequences, ['###']);
  assert.equal(genCfg.responseMimeType, 'application/json');
  const contents = body.contents as { role: string; parts: Record<string, unknown>[] }[];
  assert.equal(contents[0].role, 'user');
  assert.deepEqual(contents[0].parts[1], { inlineData: { mimeType: 'image/png', data: 'AAAA' } });
});

test('OpenAI streaming request injects stream_options.include_usage', () => {
  const req = parseClientRequest('openai', {
    model: 'auto',
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
  });
  const { path, body } = buildUpstreamRequest(req, 'openai', 'z-ai/glm-5.3-flash:free');
  assert.equal(path, '/v1/chat/completions');
  assert.deepEqual(body.stream_options, { include_usage: true });
});

test('Anthropic SSE rendering preserves event order, usage, and tool_use blocks', () => {
  const sse = renderClientSse('anthropic', 'auto', {
    text: 'Calling lookup',
    toolCalls: [{ id: 'toolu_1', name: 'lookup', argumentsJson: '{"id":"42"}', argumentsObj: { id: '42' } }],
    stopReason: 'tool_calls',
    usage: { input_tokens: 18, output_tokens: 9 },
  });
  assert.match(sse, /^event: message_start\n/);
  assert.match(sse, /"input_tokens":18/);
  assert.match(sse, /event: content_block_delta/);
  assert.match(sse, /"type":"input_json_delta"/);
  assert.match(sse, /"stop_reason":"tool_use"/);
  assert.match(sse, /"output_tokens":9/);
  assert.match(sse, /event: message_stop\n/);
});

test('OpenAI JSON rendering preserves exact usage and tool_calls', () => {
  const json = renderClientJson('openai', 'auto', {
    text: 'Done',
    toolCalls: [],
    stopReason: 'stop',
    usage: { input_tokens: 11, output_tokens: 4 },
  }) as Record<string, unknown>;
  assert.equal(json.id, 'chatcmpl-sim');
  assert.equal(json.created, 1767225600);
  assert.deepEqual(json.usage, { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 });
});

test('Router excludes anthropic wire for OpenAI json_mode and preserves scarce vision slots', () => {
  const gemini = makeProvider('gemini', 'gemini');
  const zai = makeProvider('zai', 'anthropic');
  const cf = makeProvider('cloudflare', 'openai');

  const brief: Brief = {
    directory: { schema: 'free-model-aggregator-01/providers/v1', providers: [gemini, zai, cf] },
    ratings: {
      schema: 'free-model-aggregator-01/ratings/v1',
      best: [],
      ratings: [
        { provider: 'gemini', model: 'gemini-3.8-flash', rating_milli: 1458, context_length: 1000000, tools: true, vision: true, json: true, free: true },
        { provider: 'zai', model: 'glm-4.7-flash', rating_milli: 1470, context_length: 204800, tools: true, vision: false, json: true, free: true },
        { provider: 'cloudflare', model: 'glm-4.7-flash', rating_milli: 1470, context_length: 204800, tools: true, vision: false, json: true, free: true },
      ],
    },
    workload: {
      schema: 'free-model-aggregator-01/workload/v1',
      manifest: [
        { id: 'r01', api: 'openai', stream: false, weight: 1, requires: { json: true } },
        { id: 'r02', api: 'openai', stream: false, weight: 2, requires: { vision: true, tools: true } },
      ],
    },
    error: null,
  };

  const probes: Probe[] = brief.directory.providers.map((provider) => ({
    provider,
    ok: true,
    status: 200,
    models: brief.ratings!.ratings
      .filter((r) => r.provider === provider.id)
      .map((r) => ({ id: r.model, display_name: r.model, free: true, context_length: r.context_length, tools: r.tools, vision: r.vision, json: r.json })),
    error: null,
  }));

  const state = createRouterState(brief, probes);
  const pick1 = selectNextCandidate(
    state,
    { clientWire: 'openai', stream: false, tools: false, vision: false, json: true, strictJsonWire: true, minContext: 0 },
    new Set(),
  );
  assert.ok(pick1 !== null);
  assert.notEqual(pick1.provider.id, 'zai');
  recordProviderOutcome(state, pick1.provider.id, true, 200, true);

  const pick2 = selectNextCandidate(
    state,
    { clientWire: 'openai', stream: false, tools: true, vision: true, json: false, strictJsonWire: false, minContext: 0 },
    new Set(),
  );
  assert.equal(pick2?.provider.id, 'gemini');
});

test('Model list filters paid models and deduplicates free models deterministically', () => {
  const catalog: CatalogReport = {
    providers: [
      {
        id: 'openrouter',
        base_url: '/p/openrouter',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: [
          { id: 'free-b', free: true, context_length: 128000 },
          { id: 'paid-x', free: false, context_length: 200000 },
          { id: 'free-a', free: true, context_length: 128000 },
        ],
      },
    ],
  };
  const oa = openAIModelList(catalog) as { data: { id: string }[] };
  const an = anthropicModelList(catalog) as { data: { id: string }[] };
  assert.deepEqual(oa.data.map((m) => m.id), ['free-a', 'free-b']);
  assert.deepEqual(an.data.map((m) => m.id), ['free-a', 'free-b']);
});

test('Router prefers untouched Call #1 providers within loss margin and permanently avoids 402 errors', () => {
  const kilol = makeProvider('kilol', 'openai');
  const groq = makeProvider('groq', 'openai');
  const sambanova = makeProvider('sambanova', 'openai');
  const gemini = makeProvider('gemini', 'gemini');

  const brief: Brief = {
    directory: { schema: 'free-model-aggregator-01/providers/v1', providers: [kilol, groq, sambanova, gemini] },
    ratings: {
      schema: 'free-model-aggregator-01/ratings/v1',
      best: [
        { id: 'r01', rating_milli: 1475 },
        { id: 'r02', rating_milli: 1475 },
      ],
      ratings: [
        { provider: 'kilol', model: 'glm-5.3-flash', rating_milli: 1475, context_length: 204800, tools: true, vision: false, json: true, free: true },
        { provider: 'groq', model: 'glm-4.7-flash', rating_milli: 1470, context_length: 204800, tools: true, vision: false, json: true, free: true },
        { provider: 'sambanova', model: 'qwen3.5-397b', rating_milli: 1440, context_length: 204800, tools: true, vision: false, json: true, free: true },
        { provider: 'gemini', model: 'gemini-3.8-flash', rating_milli: 1458, context_length: 1000000, tools: true, vision: true, json: true, free: true },
      ],
    },
    workload: {
      schema: 'free-model-aggregator-01/workload/v1',
      manifest: [
        { id: 'r01', api: 'openai', stream: false, weight: 1, requires: {} },
        { id: 'r02', api: 'openai', stream: false, weight: 1, requires: {} },
      ],
    },
    error: null,
  };

  const probes: Probe[] = brief.directory.providers.map((provider) => ({
    provider,
    ok: true,
    status: 200,
    models: brief.ratings!.ratings
      .filter((r) => r.provider === provider.id)
      .map((r) => ({ id: r.model, display_name: r.model, free: true, context_length: r.context_length, tools: r.tools, vision: r.vision, json: r.json })),
    error: null,
  }));

  const state = createRouterState(brief, probes);
  const firstPick = selectNextCandidate(
    state,
    { clientWire: 'openai', stream: false, tools: false, vision: false, json: false, strictJsonWire: false, minContext: 0 },
    new Set(),
  );
  assert.equal(firstPick?.provider.id, 'kilol');
  recordProviderOutcome(state, 'kilol', true, 200, true, firstPick!.ratingMilli);

  // On r02, kilol already has calls=1 so its Call #2 carries hazard risk; untouched groq (calls=0, 1470) wins cleanly.
  const secondPick = selectNextCandidate(
    state,
    { clientWire: 'openai', stream: false, tools: false, vision: false, json: false, strictJsonWire: false, minContext: 0 },
    new Set(),
  );
  assert.equal(secondPick?.provider.id, 'groq');

  // If groq returns 402, router avoids groq afterward and picks next untouched healthy candidate (gemini 1458).
  recordProviderOutcome(state, 'groq', false, 402, false);
  const recoveryPick = selectNextCandidate(
    state,
    { clientWire: 'openai', stream: false, tools: false, vision: false, json: false, strictJsonWire: false, minContext: 0 },
    new Set(['groq']),
  );
  assert.equal(recoveryPick?.provider.id, 'gemini');
});
