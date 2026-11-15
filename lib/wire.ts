// Protocol translation between downstream OpenAI/Anthropic clients and upstream OpenAI/Anthropic/Gemini providers.
// Everything here is deterministic (fixed synthetic ids and timestamps) with zero wall-clock or PRNG calls.

import { isRecord, type CatalogReport, type Json, type Wire } from './providers.ts';

export type ClientWire = 'openai' | 'anthropic';
export type StopReason = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CanonicalPart =
  | { type: 'text'; text: string }
  | { type: 'image_base64'; media_type: string; data: string }
  | { type: 'image_url'; url: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; name?: string; content: string };

export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: CanonicalPart[];
  tool_call_id?: string;
  name?: string;
}

export interface CanonicalRequest {
  clientWire: ClientWire;
  clientModel: string;
  system: string | null;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  tool_choice?: unknown;
  max_tokens?: number;
  stop?: string[];
  temperature?: number;
  top_p?: number;
  json_mode: boolean;
  response_format?: Json;
  stream: boolean;
}

export interface CanonicalToolCall {
  id: string;
  name: string;
  argumentsJson: string;
  argumentsObj: unknown;
}

export interface CanonicalResponse {
  text: string;
  toolCalls: CanonicalToolCall[];
  stopReason: StopReason;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface TranslatedUpstreamResult {
  ok: boolean;
  status: number;
  response?: CanonicalResponse;
  error?: string;
}

const CREATED_EPOCH = 1767225600;
const CREATED_ISO = '2026-01-01T00:00:00Z';

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumber(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function parseDataUrl(url: string): { media_type: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.+)$/s.exec(url);
  if (!match) return null;
  return { media_type: match[1], data: match[2] };
}

function contentToText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const chunks: string[] = [];
  for (const item of content) {
    if (typeof item === 'string') {
      chunks.push(item);
    } else if (isRecord(item) && typeof item.text === 'string') {
      chunks.push(item.text);
    } else if (isRecord(item) && item.type === 'tool_result') {
      const inner = contentToText(item.content);
      if (inner !== '') chunks.push(inner);
    }
  }
  return chunks.join('\n');
}

function parseOpenAIContentParts(msg: Json): CanonicalPart[] {
  const parts: CanonicalPart[] = [];
  const raw = msg.content;
  if (typeof raw === 'string' && raw !== '') {
    parts.push({ type: 'text', text: raw });
  } else if (Array.isArray(raw)) {
    for (const block of raw) {
      if (typeof block === 'string') {
        parts.push({ type: 'text', text: block });
      } else if (isRecord(block)) {
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image_url') {
          const url = isRecord(block.image_url) ? asString(block.image_url.url) : asString(block.image_url);
          if (url !== '') {
            const parsed = parseDataUrl(url);
            if (parsed !== null) {
              parts.push({ type: 'image_base64', media_type: parsed.media_type, data: parsed.data });
            } else {
              parts.push({ type: 'image_url', url });
            }
          }
        }
      }
    }
  }
  if (Array.isArray(msg.tool_calls)) {
    for (let i = 0; i < msg.tool_calls.length; i++) {
      const tc = msg.tool_calls[i];
      if (!isRecord(tc)) continue;
      const fn = isRecord(tc.function) ? tc.function : {};
      const name = asString(fn.name);
      const argsRaw = fn.arguments;
      let input: unknown = {};
      if (typeof argsRaw === 'string' && argsRaw.trim() !== '') {
        try {
          input = JSON.parse(argsRaw);
        } catch {
          input = { raw: argsRaw };
        }
      } else if (isRecord(argsRaw)) {
        input = argsRaw;
      }
      parts.push({
        type: 'tool_use',
        id: asString(tc.id) || `call_sim_${i}`,
        name,
        input,
      });
    }
  }
  if (parts.length === 0) parts.push({ type: 'text', text: '' });
  return parts;
}

function parseAnthropicContentParts(msg: Json): CanonicalPart[] {
  const parts: CanonicalPart[] = [];
  const raw = msg.content;
  if (typeof raw === 'string') {
    parts.push({ type: 'text', text: raw });
    return parts;
  }
  if (!Array.isArray(raw)) {
    parts.push({ type: 'text', text: '' });
    return parts;
  }
  for (const block of raw) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ type: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({
          type: 'image_base64',
          media_type: asString(block.source.media_type) || 'image/png',
          data: block.source.data,
        });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ type: 'image_url', url: block.source.url });
      }
    } else if (block.type === 'tool_use') {
      parts.push({
        type: 'tool_use',
        id: asString(block.id) || 'call_sim_0',
        name: asString(block.name),
        input: block.input ?? {},
      });
    } else if (block.type === 'tool_result') {
      parts.push({
        type: 'tool_result',
        tool_use_id: asString(block.tool_use_id) || 'call_sim_0',
        content: contentToText(block.content),
      });
    }
  }
  if (parts.length === 0) parts.push({ type: 'text', text: '' });
  return parts;
}

/** Parses an incoming OpenAI or Anthropic request body into a canonical representation. */
export function parseClientRequest(clientWire: ClientWire, body: Json): CanonicalRequest {
  const systemParts: string[] = [];
  const messages: CanonicalMessage[] = [];

  if (clientWire === 'openai') {
    const rawMsgs = Array.isArray(body.messages) ? body.messages : [];
    for (const item of rawMsgs) {
      if (!isRecord(item)) continue;
      const role = asString(item.role);
      if (role === 'system' || role === 'developer') {
        const txt = contentToText(item.content);
        if (txt !== '') systemParts.push(txt);
      } else if (role === 'tool') {
        messages.push({
          role: 'tool',
          tool_call_id: asString(item.tool_call_id) || 'call_sim_0',
          name: asString(item.name) || undefined,
          parts: [{ type: 'tool_result', tool_use_id: asString(item.tool_call_id) || 'call_sim_0', content: contentToText(item.content) }],
        });
      } else {
        messages.push({
          role: role === 'assistant' ? 'assistant' : 'user',
          parts: parseOpenAIContentParts(item),
        });
      }
    }
  } else {
    const sys = contentToText(body.system);
    if (sys !== '') systemParts.push(sys);
    const rawMsgs = Array.isArray(body.messages) ? body.messages : [];
    for (const item of rawMsgs) {
      if (!isRecord(item)) continue;
      const role = item.role === 'assistant' ? 'assistant' : 'user';
      messages.push({ role, parts: parseAnthropicContentParts(item) });
    }
  }

  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      if (t.type === 'function' && isRecord(t.function)) {
        const name = asString(t.function.name);
        if (name === '') continue;
        tools.push({
          name,
          description: asString(t.function.description),
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
        });
      } else if (typeof t.name === 'string' && t.name !== '') {
        tools.push({
          name: t.name,
          description: asString(t.description),
          parameters: isRecord(t.input_schema)
            ? t.input_schema
            : isRecord(t.parameters)
              ? t.parameters
              : { type: 'object', properties: {} },
        });
      }
    }
  }

  const rawStop = clientWire === 'openai' ? body.stop : body.stop_sequences;
  let stop: string[] | undefined;
  if (typeof rawStop === 'string' && rawStop !== '') {
    stop = [rawStop];
  } else if (Array.isArray(rawStop)) {
    const filtered = rawStop.filter((s): s is string => typeof s === 'string');
    if (filtered.length > 0) stop = filtered;
  }

  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  const responseFormat = isRecord(body.response_format) ? body.response_format : undefined;
  const rfType = responseFormat ? asString(responseFormat.type) : '';
  const jsonMode = rfType === 'json_object' || rfType === 'json_schema';

  return {
    clientWire,
    clientModel: asString(body.model) || 'auto',
    system: systemParts.length > 0 ? systemParts.join('\n') : null,
    messages,
    tools,
    tool_choice: body.tool_choice,
    max_tokens: maxTokens,
    stop,
    temperature: asNumber(body.temperature),
    top_p: asNumber(body.top_p),
    json_mode: jsonMode,
    response_format: responseFormat,
    stream: body.stream === true,
  };
}

/** Detects capability/wire requirements directly from a parsed client request. */
export function detectRequestNeeds(req: CanonicalRequest): {
  tools: boolean;
  vision: boolean;
  json: boolean;
  strictJsonWire: boolean;
  minContext: number;
} {
  let vision = false;
  let totalChars = req.system ? req.system.length : 0;
  for (const m of req.messages) {
    for (const p of m.parts) {
      if (p.type === 'image_base64' || p.type === 'image_url') vision = true;
      else if (p.type === 'text') totalChars += p.text.length;
      else if (p.type === 'tool_result') totalChars += p.content.length;
      else if (p.type === 'tool_use') totalChars += JSON.stringify(p.input ?? {}).length;
    }
  }
  const approxTokens = Math.ceil(totalChars / 4);
  return {
    tools: req.tools.length > 0,
    vision,
    json: req.json_mode,
    strictJsonWire: req.clientWire === 'openai' && req.json_mode,
    minContext: approxTokens,
  };
}

function renderOpenAIMessages(req: CanonicalRequest): Json[] {
  const out: Json[] = [];
  if (req.system !== null && req.system !== '') {
    out.push({ role: 'system', content: req.system });
  }
  for (const m of req.messages) {
    if (m.role === 'tool') {
      const tr = m.parts.find((p) => p.type === 'tool_result');
      out.push({
        role: 'tool',
        tool_call_id: m.tool_call_id ?? (tr && tr.type === 'tool_result' ? tr.tool_use_id : 'call_sim_0'),
        content: tr && tr.type === 'tool_result' ? tr.content : '',
      });
      continue;
    }

    const toolResults = m.parts.filter((p): p is Extract<CanonicalPart, { type: 'tool_result' }> => p.type === 'tool_result');
    if (toolResults.length > 0 && m.role === 'user') {
      const nonResults = m.parts.filter((p) => p.type !== 'tool_result');
      for (const tr of toolResults) {
        out.push({ role: 'tool', tool_call_id: tr.tool_use_id, content: tr.content });
      }
      if (nonResults.length === 0) continue;
    }

    const toolUses = m.parts.filter((p): p is Extract<CanonicalPart, { type: 'tool_use' }> => p.type === 'tool_use');
    const otherParts = m.parts.filter((p) => p.type !== 'tool_use' && p.type !== 'tool_result');

    if (m.role === 'assistant' && toolUses.length > 0) {
      const text = otherParts
        .filter((p): p is Extract<CanonicalPart, { type: 'text' }> => p.type === 'text')
        .map((p) => p.text)
        .join('\n');
      out.push({
        role: 'assistant',
        content: text === '' ? null : text,
        tool_calls: toolUses.map((tu) => ({
          id: tu.id,
          type: 'function',
          function: {
            name: tu.name,
            arguments: typeof tu.input === 'string' ? tu.input : JSON.stringify(tu.input ?? {}),
          },
        })),
      });
      continue;
    }

    const hasMedia = otherParts.some((p) => p.type === 'image_base64' || p.type === 'image_url');
    if (!hasMedia && otherParts.length === 1 && otherParts[0].type === 'text') {
      out.push({ role: m.role, content: otherParts[0].text });
    } else if (!hasMedia) {
      const text = otherParts
        .filter((p): p is Extract<CanonicalPart, { type: 'text' }> => p.type === 'text')
        .map((p) => p.text)
        .join('\n');
      out.push({ role: m.role, content: text });
    } else {
      const blocks: Json[] = [];
      for (const p of otherParts) {
        if (p.type === 'text') {
          blocks.push({ type: 'text', text: p.text });
        } else if (p.type === 'image_base64') {
          blocks.push({ type: 'image_url', image_url: { url: `data:${p.media_type};base64,${p.data}` } });
        } else if (p.type === 'image_url') {
          blocks.push({ type: 'image_url', image_url: { url: p.url } });
        }
      }
      out.push({ role: m.role, content: blocks });
    }
  }
  return out;
}

function renderAnthropicMessages(req: CanonicalRequest): Json[] {
  const out: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'tool') {
      const tr = m.parts.find((p) => p.type === 'tool_result');
      out.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: m.tool_call_id ?? (tr && tr.type === 'tool_result' ? tr.tool_use_id : 'call_sim_0'),
            content: tr && tr.type === 'tool_result' ? tr.content : '',
          },
        ],
      });
      continue;
    }

    const onlyOneText = m.parts.length === 1 && m.parts[0].type === 'text';
    if (onlyOneText && m.parts[0].type === 'text') {
      out.push({ role: m.role, content: m.parts[0].text });
      continue;
    }

    const blocks: Json[] = [];
    for (const p of m.parts) {
      if (p.type === 'text') {
        blocks.push({ type: 'text', text: p.text });
      } else if (p.type === 'image_base64') {
        blocks.push({
          type: 'image',
          source: { type: 'base64', media_type: p.media_type, data: p.data },
        });
      } else if (p.type === 'image_url') {
        blocks.push({
          type: 'image',
          source: { type: 'url', url: p.url },
        });
      } else if (p.type === 'tool_use') {
        blocks.push({
          type: 'tool_use',
          id: p.id,
          name: p.name,
          input: isRecord(p.input) ? p.input : {},
        });
      } else if (p.type === 'tool_result') {
        blocks.push({
          type: 'tool_result',
          tool_use_id: p.tool_use_id,
          content: p.content,
        });
      }
    }
    out.push({ role: m.role, content: blocks });
  }
  return out;
}

function renderGeminiBody(req: CanonicalRequest): Json {
  const contents: Json[] = [];
  for (const m of req.messages) {
    const parts: Json[] = [];
    for (const p of m.parts) {
      if (p.type === 'text') {
        parts.push({ text: p.text });
      } else if (p.type === 'image_base64') {
        parts.push({ inlineData: { mimeType: p.media_type, data: p.data } });
      } else if (p.type === 'image_url') {
        parts.push({ fileData: { fileUri: p.url } });
      } else if (p.type === 'tool_use') {
        parts.push({ functionCall: { name: p.name, args: isRecord(p.input) ? p.input : {} } });
      } else if (p.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: m.name ?? p.name ?? 'tool',
            response: { content: p.content },
          },
        });
      }
    }
    contents.push({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts,
    });
  }

  const body: Json = { contents };
  if (req.system !== null && req.system !== '') {
    body.systemInstruction = { parts: [{ text: req.system }] };
  }
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const genCfg: Json = {};
  if (req.max_tokens !== undefined) genCfg.maxOutputTokens = req.max_tokens;
  if (req.stop !== undefined && req.stop.length > 0) genCfg.stopSequences = req.stop;
  if (req.temperature !== undefined) genCfg.temperature = req.temperature;
  if (req.top_p !== undefined) genCfg.topP = req.top_p;
  if (req.json_mode) genCfg.responseMimeType = 'application/json';
  if (Object.keys(genCfg).length > 0) body.generationConfig = genCfg;

  return body;
}

/** Builds an upstream URL path and JSON body matching the target provider's wire protocol. */
export function buildUpstreamRequest(
  req: CanonicalRequest,
  upstreamWire: Wire,
  upstreamModelId: string,
): { path: string; body: Json } {
  if (upstreamWire === 'openai') {
    const body: Json = {
      model: upstreamModelId,
      messages: renderOpenAIMessages(req),
      stream: req.stream,
    };
    if (req.stream) body.stream_options = { include_usage: true };
    if (req.max_tokens !== undefined) body.max_tokens = req.max_tokens;
    if (req.stop !== undefined && req.stop.length > 0) body.stop = req.stop;
    if (req.temperature !== undefined) body.temperature = req.temperature;
    if (req.top_p !== undefined) body.top_p = req.top_p;
    if (req.json_mode) body.response_format = req.response_format ?? { type: 'json_object' };
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
      if (req.tool_choice !== undefined) body.tool_choice = req.tool_choice;
    }
    return { path: '/v1/chat/completions', body };
  }

  if (upstreamWire === 'anthropic') {
    const body: Json = {
      model: upstreamModelId,
      messages: renderAnthropicMessages(req),
      stream: req.stream,
    };
    if (req.max_tokens !== undefined) body.max_tokens = req.max_tokens;
    if (req.system !== null && req.system !== '') body.system = req.system;
    if (req.stop !== undefined && req.stop.length > 0) body.stop_sequences = req.stop;
    if (req.temperature !== undefined) body.temperature = req.temperature;
    if (req.top_p !== undefined) body.top_p = req.top_p;
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters,
      }));
    }
    return { path: '/v1/messages', body };
  }

  const encodedModel = encodeURIComponent(upstreamModelId);
  const path = req.stream
    ? `/v1beta/models/${encodedModel}:streamGenerateContent?alt=sse`
    : `/v1beta/models/${encodedModel}:generateContent`;
  return { path, body: renderGeminiBody(req) };
}

function mapStopReason(raw: unknown, hasTools: boolean): StopReason {
  if (typeof raw === 'string') {
    if (raw === 'length' || raw === 'max_tokens' || raw === 'MAX_TOKENS') return 'length';
    if (raw === 'tool_calls' || raw === 'tool_use') return 'tool_calls';
    if (raw === 'content_filter' || raw === 'refusal' || raw === 'SAFETY' || raw === 'RECITATION') {
      return 'content_filter';
    }
    if (raw === 'stop' || raw === 'end_turn' || raw === 'STOP') {
      return hasTools ? 'tool_calls' : 'stop';
    }
  }
  return hasTools ? 'tool_calls' : 'stop';
}

function parseJsonResponse(wire: Wire, body: unknown): CanonicalResponse | null {
  if (!isRecord(body)) return null;

  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const first = isRecord(choices[0]) ? choices[0] : null;
    if (first === null) return null;
    const msg = isRecord(first.message) ? first.message : {};
    const text = typeof msg.content === 'string' ? msg.content : '';
    const toolCalls: CanonicalToolCall[] = [];
    if (Array.isArray(msg.tool_calls)) {
      for (let i = 0; i < msg.tool_calls.length; i++) {
        const tc = msg.tool_calls[i];
        if (!isRecord(tc) || !isRecord(tc.function)) continue;
        const name = asString(tc.function.name);
        if (name === '') continue;
        const rawArgs = typeof tc.function.arguments === 'string'
          ? tc.function.arguments
          : JSON.stringify(tc.function.arguments ?? {});
        let parsed: unknown = {};
        try {
          parsed = JSON.parse(rawArgs);
        } catch {
          parsed = {};
        }
        toolCalls.push({ id: asString(tc.id) || `call_sim_${i}`, name, argumentsJson: rawArgs, argumentsObj: parsed });
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      toolCalls,
      stopReason: mapStopReason(first.finish_reason, toolCalls.length > 0),
      usage: {
        input_tokens: asNumber(usage.prompt_tokens) ?? 0,
        output_tokens: asNumber(usage.completion_tokens) ?? 0,
      },
    };
  }

  if (wire === 'anthropic') {
    const blocks = Array.isArray(body.content) ? body.content : [];
    const textParts: string[] = [];
    const toolCalls: CanonicalToolCall[] = [];
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') textParts.push(b.text);
      else if (b.type === 'tool_use') {
        const name = asString(b.name);
        const input = b.input ?? {};
        toolCalls.push({
          id: asString(b.id) || `toolu_sim_${i}`,
          name,
          argumentsJson: JSON.stringify(input),
          argumentsObj: input,
        });
      }
    }
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: textParts.join(''),
      toolCalls,
      stopReason: mapStopReason(body.stop_reason, toolCalls.length > 0),
      usage: {
        input_tokens: asNumber(usage.input_tokens) ?? 0,
        output_tokens: asNumber(usage.output_tokens) ?? 0,
      },
    };
  }

  // Gemini JSON
  const cands = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(cands[0]) ? cands[0] : null;
  if (first === null) return null;
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const textParts: string[] = [];
  const toolCalls: CanonicalToolCall[] = [];
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') textParts.push(p.text);
    if (isRecord(p.functionCall)) {
      const name = asString(p.functionCall.name);
      const args = p.functionCall.args ?? {};
      toolCalls.push({
        id: `call_sim_${i}`,
        name,
        argumentsJson: JSON.stringify(args),
        argumentsObj: args,
      });
    }
  }
  const usageMeta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: textParts.join(''),
    toolCalls,
    stopReason: mapStopReason(first.finishReason, toolCalls.length > 0),
    usage: {
      input_tokens: asNumber(usageMeta.promptTokenCount) ?? 0,
      output_tokens: asNumber(usageMeta.candidatesTokenCount) ?? 0,
    },
  };
}

interface ParsedSseFrame {
  event: string;
  data: string;
}

function parseSseFrames(rawText: string): ParsedSseFrame[] {
  const frames: ParsedSseFrame[] = [];
  const normalized = rawText.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event = '';
    const dataLines: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

function parseSseResponse(wire: Wire, rawText: string): CanonicalResponse | null {
  const frames = parseSseFrames(rawText);
  if (frames.length === 0) return null;

  if (wire === 'openai') {
    let text = '';
    let finishRaw: string | undefined;
    let inTok = 0;
    let outTok = 0;
    const tcMap = new Map<number, { id: string; name: string; args: string }>();
    let sawPayload = false;

    for (const frame of frames) {
      if (frame.data === '[DONE]') continue;
      let obj: unknown;
      try {
        obj = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(obj)) continue;
      sawPayload = true;
      if (isRecord(obj.usage)) {
        inTok = asNumber(obj.usage.prompt_tokens) ?? inTok;
        outTok = asNumber(obj.usage.completion_tokens) ?? outTok;
      }
      const choices = Array.isArray(obj.choices) ? obj.choices : [];
      const first = isRecord(choices[0]) ? choices[0] : null;
      if (first === null) continue;
      if (typeof first.finish_reason === 'string' && first.finish_reason !== '') finishRaw = first.finish_reason;
      const delta = isRecord(first.delta) ? first.delta : null;
      if (delta === null) continue;
      if (typeof delta.content === 'string') text += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const item of delta.tool_calls) {
          if (!isRecord(item)) continue;
          const idx = asNumber(item.index) ?? 0;
          const entry = tcMap.get(idx) ?? { id: `call_sim_${idx}`, name: '', args: '' };
          if (typeof item.id === 'string' && item.id !== '') entry.id = item.id;
          if (isRecord(item.function)) {
            if (typeof item.function.name === 'string') entry.name += item.function.name;
            if (typeof item.function.arguments === 'string') entry.args += item.function.arguments;
          }
          tcMap.set(idx, entry);
        }
      }
    }
    if (!sawPayload) return null;
    const toolCalls: CanonicalToolCall[] = [...tcMap.entries()]
      .sort((a, b) => a[0] - b[0])
      .filter(([, tc]) => tc.name !== '')
      .map(([, tc]) => {
        let parsed: unknown = {};
        try {
          parsed = JSON.parse(tc.args);
        } catch {
          parsed = {};
        }
        return { id: tc.id, name: tc.name, argumentsJson: tc.args || '{}', argumentsObj: parsed };
      });
    return {
      text,
      toolCalls,
      stopReason: mapStopReason(finishRaw, toolCalls.length > 0),
      usage: { input_tokens: inTok, output_tokens: outTok },
    };
  }

  if (wire === 'anthropic') {
    let inTok = 0;
    let outTok = 0;
    let stopRaw: string | undefined;
    const blockMap = new Map<number, { type: string; text: string; id: string; name: string; args: string }>();
    let sawPayload = false;

    for (const frame of frames) {
      let obj: unknown;
      try {
        obj = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(obj)) continue;
      sawPayload = true;
      const evType = asString(obj.type) || frame.event;
      if (evType === 'message_start' && isRecord(obj.message) && isRecord(obj.message.usage)) {
        inTok = asNumber(obj.message.usage.input_tokens) ?? inTok;
        outTok = asNumber(obj.message.usage.output_tokens) ?? outTok;
      } else if (evType === 'content_block_start' && isRecord(obj.content_block)) {
        const idx = asNumber(obj.index) ?? blockMap.size;
        const cb = obj.content_block;
        blockMap.set(idx, {
          type: asString(cb.type),
          text: asString(cb.text),
          id: asString(cb.id) || `toolu_sim_${idx}`,
          name: asString(cb.name),
          args: cb.type === 'tool_use' && isRecord(cb.input) && Object.keys(cb.input).length > 0 ? JSON.stringify(cb.input) : '',
        });
      } else if (evType === 'content_block_delta' && isRecord(obj.delta)) {
        const idx = asNumber(obj.index) ?? 0;
        const entry = blockMap.get(idx) ?? { type: 'text', text: '', id: `toolu_sim_${idx}`, name: '', args: '' };
        if (obj.delta.type === 'text_delta' && typeof obj.delta.text === 'string') entry.text += obj.delta.text;
        else if (obj.delta.type === 'input_json_delta' && typeof obj.delta.partial_json === 'string') {
          entry.args += obj.delta.partial_json;
        }
        blockMap.set(idx, entry);
      } else if (evType === 'message_delta') {
        if (isRecord(obj.delta) && typeof obj.delta.stop_reason === 'string') stopRaw = obj.delta.stop_reason;
        if (isRecord(obj.usage)) outTok = asNumber(obj.usage.output_tokens) ?? outTok;
      }
    }
    if (!sawPayload) return null;
    const textParts: string[] = [];
    const toolCalls: CanonicalToolCall[] = [];
    for (const [, block] of [...blockMap.entries()].sort((a, b) => a[0] - b[0])) {
      if (block.type === 'text') textParts.push(block.text);
      else if (block.type === 'tool_use' && block.name !== '') {
        let parsed: unknown = {};
        try {
          parsed = JSON.parse(block.args || '{}');
        } catch {
          parsed = {};
        }
        toolCalls.push({
          id: block.id,
          name: block.name,
          argumentsJson: block.args || '{}',
          argumentsObj: parsed,
        });
      }
    }
    return {
      text: textParts.join(''),
      toolCalls,
      stopReason: mapStopReason(stopRaw, toolCalls.length > 0),
      usage: { input_tokens: inTok, output_tokens: outTok },
    };
  }

  // Gemini SSE
  const textParts: string[] = [];
  const toolCalls: CanonicalToolCall[] = [];
  let finishRaw: string | undefined;
  let inTok = 0;
  let outTok = 0;
  let sawPayload = false;

  for (const frame of frames) {
    let obj: unknown;
    try {
      obj = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(obj)) continue;
    sawPayload = true;
    if (isRecord(obj.usageMetadata)) {
      inTok = asNumber(obj.usageMetadata.promptTokenCount) ?? inTok;
      outTok = asNumber(obj.usageMetadata.candidatesTokenCount) ?? outTok;
    }
    const cands = Array.isArray(obj.candidates) ? obj.candidates : [];
    const first = isRecord(cands[0]) ? cands[0] : null;
    if (first === null) continue;
    if (typeof first.finishReason === 'string' && first.finishReason !== '') finishRaw = first.finishReason;
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (!isRecord(p)) continue;
      if (typeof p.text === 'string') textParts.push(p.text);
      if (isRecord(p.functionCall)) {
        const name = asString(p.functionCall.name);
        const args = p.functionCall.args ?? {};
        toolCalls.push({
          id: `call_sim_${toolCalls.length}`,
          name,
          argumentsJson: JSON.stringify(args),
          argumentsObj: args,
        });
      }
    }
  }
  if (!sawPayload) return null;
  return {
    text: textParts.join(''),
    toolCalls,
    stopReason: mapStopReason(finishRaw, toolCalls.length > 0),
    usage: { input_tokens: inTok, output_tokens: outTok },
  };
}

/** Reads and parses an upstream HTTP response (JSON or SSE) into canonical form. */
export async function readUpstreamResponse(wire: Wire, stream: boolean, res: Response): Promise<TranslatedUpstreamResult> {
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    return { ok: false, status: res.status, error: `HTTP ${res.status}${errText ? `: ${errText.slice(0, 160)}` : ''}` };
  }
  const raw = await res.text();
  const ctype = (res.headers.get('content-type') ?? '').toLowerCase();
  const isSse = stream || ctype.includes('text/event-stream');

  const parsed = isSse
    ? parseSseResponse(wire, raw)
    : (() => {
        try {
          return parseJsonResponse(wire, JSON.parse(raw));
        } catch {
          return null;
        }
      })();

  if (parsed === null || (parsed.text === '' && parsed.toolCalls.length === 0)) {
    return { ok: false, status: 502, error: 'Upstream returned empty or unparseable completion' };
  }
  return { ok: true, status: res.status, response: parsed };
}

/** Formats a canonical completion response as OpenAI or Anthropic JSON. */
export function renderClientJson(clientWire: ClientWire, clientModel: string, resp: CanonicalResponse): Json {
  if (clientWire === 'openai') {
    const message: Json = { role: 'assistant', content: resp.text };
    if (resp.toolCalls.length > 0) {
      message.tool_calls = resp.toolCalls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: tc.argumentsJson },
      }));
    }
    return {
      id: 'chatcmpl-sim',
      object: 'chat.completion',
      created: CREATED_EPOCH,
      model: clientModel,
      choices: [{ index: 0, message, finish_reason: resp.stopReason }],
      usage: {
        prompt_tokens: resp.usage.input_tokens,
        completion_tokens: resp.usage.output_tokens,
        total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
      },
    };
  }

  const anthropicStop =
    resp.stopReason === 'length'
      ? 'max_tokens'
      : resp.stopReason === 'tool_calls'
        ? 'tool_use'
        : resp.stopReason === 'content_filter'
          ? 'refusal'
          : 'end_turn';

  const content: Json[] = [];
  if (resp.text !== '' || resp.toolCalls.length === 0) {
    content.push({ type: 'text', text: resp.text });
  }
  for (const tc of resp.toolCalls) {
    content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.argumentsObj ?? {} });
  }
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model: clientModel,
    content,
    stop_reason: anthropicStop,
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage.input_tokens,
      output_tokens: resp.usage.output_tokens,
    },
  };
}

/** Formats a canonical completion response as an OpenAI or Anthropic SSE byte payload. */
export function renderClientSse(clientWire: ClientWire, clientModel: string, resp: CanonicalResponse): string {
  if (clientWire === 'openai') {
    const frames: string[] = [];
    const emit = (obj: Json) => frames.push(`data: ${JSON.stringify(obj)}\n\n`);

    const deltaToolCalls = resp.toolCalls.map((tc, index) => ({
      index,
      id: tc.id,
      type: 'function',
      function: { name: tc.name, arguments: tc.argumentsJson },
    }));

    const chunkSize = 120;
    if (resp.text !== '') {
      for (let i = 0; i < resp.text.length; i += chunkSize) {
        emit({
          id: 'chatcmpl-sim',
          object: 'chat.completion.chunk',
          created: CREATED_EPOCH,
          model: clientModel,
          choices: [{ index: 0, delta: i === 0 ? { role: 'assistant', content: resp.text.slice(i, i + chunkSize) } : { content: resp.text.slice(i, i + chunkSize) }, finish_reason: null }],
        });
      }
    }
    if (deltaToolCalls.length > 0) {
      emit({
        id: 'chatcmpl-sim',
        object: 'chat.completion.chunk',
        created: CREATED_EPOCH,
        model: clientModel,
        choices: [
          {
            index: 0,
            delta: resp.text === '' ? { role: 'assistant', tool_calls: deltaToolCalls } : { tool_calls: deltaToolCalls },
            finish_reason: null,
          },
        ],
      });
    }
    emit({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: CREATED_EPOCH,
      model: clientModel,
      choices: [{ index: 0, delta: {}, finish_reason: resp.stopReason }],
    });
    emit({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: CREATED_EPOCH,
      model: clientModel,
      choices: [],
      usage: {
        prompt_tokens: resp.usage.input_tokens,
        completion_tokens: resp.usage.output_tokens,
        total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
      },
    });
    frames.push('data: [DONE]\n\n');
    return frames.join('');
  }

  const anthropicStop =
    resp.stopReason === 'length'
      ? 'max_tokens'
      : resp.stopReason === 'tool_calls'
        ? 'tool_use'
        : resp.stopReason === 'content_filter'
          ? 'refusal'
          : 'end_turn';

  const frames: string[] = [];
  const emit = (event: string, data: Json) => {
    frames.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  emit('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model: clientModel,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: resp.usage.input_tokens, output_tokens: 0 },
    },
  });

  let blockIdx = 0;
  if (resp.text !== '' || resp.toolCalls.length === 0) {
    emit('content_block_start', { type: 'content_block_start', index: blockIdx, content_block: { type: 'text', text: '' } });
    const chunkSize = 120;
    for (let i = 0; i < resp.text.length; i += chunkSize) {
      emit('content_block_delta', {
        type: 'content_block_delta',
        index: blockIdx,
        delta: { type: 'text_delta', text: resp.text.slice(i, i + chunkSize) },
      });
    }
    emit('content_block_stop', { type: 'content_block_stop', index: blockIdx });
    blockIdx += 1;
  }

  for (const tc of resp.toolCalls) {
    emit('content_block_start', {
      type: 'content_block_start',
      index: blockIdx,
      content_block: { type: 'tool_use', id: tc.id, name: tc.name, input: {} },
    });
    emit('content_block_delta', {
      type: 'content_block_delta',
      index: blockIdx,
      delta: { type: 'input_json_delta', partial_json: tc.argumentsJson },
    });
    emit('content_block_stop', { type: 'content_block_stop', index: blockIdx });
    blockIdx += 1;
  }

  emit('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStop, stop_sequence: null },
    usage: { output_tokens: resp.usage.output_tokens },
  });
  emit('message_stop', { type: 'message_stop' });
  return frames.join('');
}

function uniqueFreeModels(catalog: CatalogReport): { id: string; providerId: string }[] {
  const byId = new Map<string, string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free !== true) continue;
      if (!byId.has(model.id)) byId.set(model.id, provider.id);
    }
  }
  return [...byId.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([id, providerId]) => ({ id, providerId }));
}

/** OpenAI GET /v1/models response listing the exact set of free models in the catalog. */
export function openAIModelList(catalog: CatalogReport): Json {
  return {
    object: 'list',
    data: uniqueFreeModels(catalog).map((m) => ({
      id: m.id,
      object: 'model',
      created: CREATED_EPOCH,
      owned_by: m.providerId,
    })),
  };
}

/** Anthropic GET /v1/models response listing the exact set of free models in the catalog. */
export function anthropicModelList(catalog: CatalogReport): Json {
  return {
    data: uniqueFreeModels(catalog).map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.id,
      created_at: CREATED_ISO,
    })),
    has_more: false,
  };
}
