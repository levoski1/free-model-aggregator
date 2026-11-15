// Scarcity-aware, deterministic model router and adaptive failover state machine.
// Uses a tiny fixed-order Min-Cost Max-Flow solver over remaining manifest requests so scarce
// capabilities (vision, 200k+ context, OpenAI JSON wire) and weight-2 requests preserve top-rated capacity.

import {
  capabilitiesOf,
  modelFree,
  meetsRequirements,
  ratingOfModel,
  type Brief,
  type Capabilities,
  type Probe,
  type ProviderDirectoryEntry,
  type Requirement,
  type Wire,
  type WorkloadRequest,
} from './providers.ts';
import type { ClientWire } from './wire.ts';

export interface CandidateModel {
  provider: ProviderDirectoryEntry;
  modelId: string;
  ratingMilli: number;
  caps: Capabilities;
}

export interface ProviderState {
  entry: ProviderDirectoryEntry;
  candidates: CandidateModel[];
  calls: number;
  successes: number;
  failures: number;
  consecFailures: number;
  /** Published free-tier budget bounds for risk-adjusted routing. */
  callsMin: number;
  callsMax: number;
  /** Initial conservative estimate of healthy 2xx capacity before hitting min budget. */
  initialSafeCapacity: number;
  lastErrorStatus: number | null;
}

export interface RouterState {
  providers: ProviderState[];
  workload: WorkloadRequest[];
  cursor: number;
  hasRatings: boolean;
  bestRatingByReqId: Map<string, number>;
  totalLossSoFar: number;
}

const PROVIDER_BUDGET_RANGE: Record<string, { min: number; max: number }> = {
  openrouter: { min: 10, max: 12 },
  groq: { min: 9, max: 12 },
  kilol: { min: 12, max: 16 },
  cloudflare: { min: 7, max: 8 },
  gemini: { min: 6, max: 8 },
  ovh: { min: 5, max: 6 },
  cerebras: { min: 5, max: 6 },
  zai: { min: 4, max: 5 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
};

function fallbackRating(providerId: string, modelId: string): number {
  const id = modelId.toLowerCase();
  if (id.includes('glm-5.3-flash')) return 1475;
  if (id.includes('glm-4.7-flash')) return 1470;
  if (id.includes('deepseek-v3.2')) return 1465;
  if (id.includes('deepseek-v4.1')) return 1462;
  if (id.includes('gemini-3.8-flash')) return 1458;
  if (id.includes('gemma-4-31b')) return 1452;
  if (id.includes('qwen3.5-397b')) return 1440;
  if (providerId === 'pollinations') return 1350;
  return 1400;
}

/** Builds deterministic router state from brief documents and live catalog probes. */
export function createRouterState(brief: Brief, probes: readonly Probe[]): RouterState {
  const providers: ProviderState[] = [];

  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const candidates: CandidateModel[] = [];

    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);

      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;

      const caps = capabilitiesOf(model, row);
      const ratingMilli = row?.rating_milli ?? fallbackRating(probe.provider.id, model.id);
      candidates.push({
        provider: probe.provider,
        modelId: model.id,
        ratingMilli,
        caps,
      });
    }

    if (candidates.length === 0) continue;
    candidates.sort((a, b) => {
      if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
      return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
    });

    const bounds = PROVIDER_BUDGET_RANGE[probe.provider.id] ?? { min: 6, max: 8 };
    const safeCap = probe.provider.id === 'pollinations' ? 0 : Math.max(1, bounds.min - 2);

    providers.push({
      entry: probe.provider,
      candidates,
      calls: 0,
      successes: 0,
      failures: 0,
      consecFailures: 0,
      callsMin: bounds.min,
      callsMax: bounds.max,
      initialSafeCapacity: safeCap,
      lastErrorStatus: null,
    });
  }

  const bestRatingByReqId = new Map<string, number>();
  for (const row of brief.ratings?.best ?? []) {
    bestRatingByReqId.set(row.id, row.rating_milli);
  }

  return {
    providers,
    workload: brief.workload?.manifest ?? [],
    cursor: 0,
    hasRatings: brief.ratings !== null,
    bestRatingByReqId,
    totalLossSoFar: 0,
  };
}

/**
 * Checks if a candidate satisfies both capability requirements and wire fidelity rules.
 * Critical wire rule: simulated Anthropic wire cannot echo `json_mode: true`, so OpenAI clients
 * sending `response_format` must not be routed to `wire === 'anthropic'`.
 */
function candidateEligible(
  cand: CandidateModel,
  requires: Requirement,
  strictOpenAIJsonWire: boolean,
): boolean {
  if (strictOpenAIJsonWire && cand.provider.wire === 'anthropic') return false;
  return meetsRequirements(requires, cand.caps);
}

function bestCandidateOnProvider(
  ps: ProviderState,
  requires: Requirement,
  strictOpenAIJsonWire: boolean,
): CandidateModel | null {
  for (const c of ps.candidates) {
    if (candidateEligible(c, requires, strictOpenAIJsonWire)) return c;
  }
  return null;
}

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

const MAX_ALLOWED_LOSS = 680;
const BASE_RISK_SCALE = 140;
const FAIL1_PENALTY_RELAXED = 45;
const FAIL1_PENALTY_TIGHT = 40;

/** Solves minimum-cost flow on the small bipartite assignment graph (remaining requests x providers). */
function solveAssignment(
  providers: readonly ProviderState[],
  remaining: readonly { requires: Requirement; weight: number; strictJsonWire: boolean }[],
  excludedProviderIds: ReadonlySet<string>,
  marginLeft: number,
): Map<number, { providerIdx: number; candidate: CandidateModel }> {
  const m = remaining.length;
  const pCount = providers.length;
  const SRC = 0;
  const SINK = 1 + m + pCount;
  const nNodes = SINK + 1;

  const effRiskScale =
    marginLeft <= 150
      ? 0
      : marginLeft >= 350
        ? BASE_RISK_SCALE
        : Math.round((BASE_RISK_SCALE * (marginLeft - 150)) / 200);
  const pen1 = marginLeft > 220 ? FAIL1_PENALTY_RELAXED : FAIL1_PENALTY_TIGHT;

  const adj: FlowEdge[][] = Array.from({ length: nNodes }, () => []);
  const addEdge = (from: number, to: number, cap: number, cost: number): number => {
    const fwdIdx = adj[from].length;
    const revIdx = adj[to].length;
    adj[from].push({ to, rev: revIdx, cap, cost });
    adj[to].push({ to: from, rev: fwdIdx, cap: 0, cost: -cost });
    return fwdIdx;
  };

  // Remember edge index from req_0 to each provider so we can inspect the chosen partner.
  const reqEdges: { providerIdx: number; edgeIdx: number; cand: CandidateModel }[][] = Array.from(
    { length: m },
    () => [],
  );

  for (let r = 0; r < m; r++) {
    const reqNode = 1 + r;
    const item = remaining[r];
    addEdge(SRC, reqNode, 1, 0);

    for (let p = 0; p < pCount; p++) {
      const ps = providers[p];
      if (r === 0 && excludedProviderIds.has(ps.entry.id)) continue;
      const cand = bestCandidateOnProvider(ps, item.requires, item.strictJsonWire);
      if (cand === null) continue;

      const provNode = 1 + m + p;
      // Reward high rating weighted by request weight. Negative cost = maximize rating.
      const edgeCost = -item.weight * cand.ratingMilli;
      const eIdx = addEdge(reqNode, provNode, 1, edgeCost);
      reqEdges[r].push({ providerIdx: p, edgeIdx: eIdx, cand });
    }

    // Fallback sink edge so flow is always m even if capacities look exhausted.
    addEdge(reqNode, SINK, 1, 10_000_000);
  }

  for (let p = 0; p < pCount; p++) {
    const ps = providers[p];
    const provNode = 1 + m + p;

    // Hard avoid 402-returning or >=2 consecutive failure providers unless no viable alternative exists.
    if (ps.lastErrorStatus === 402 || ps.consecFailures >= 2) {
      addEdge(provNode, SINK, m, 5_000_000);
      continue;
    }

    if (ps.consecFailures === 1) {
      addEdge(provNode, SINK, m, pen1);
      continue;
    }

    const remSafe = Math.max(0, ps.initialSafeCapacity - ps.successes);
    let safeSlotsAdded = 0;

    // Call #1 on any healthy non-pollinations provider has 0% failure probability (fail_calls >= 2).
    if (ps.calls === 0 && ps.entry.id !== 'pollinations') {
      addEdge(provNode, SINK, 1, 0);
      safeSlotsAdded = 1;
    }

    const remAfterFirst = Math.max(0, remSafe - safeSlotsAdded);
    const midBudget = (ps.callsMin + ps.callsMax) / 2;
    for (let k = 1; k <= remAfterFirst; k++) {
      const projectedCall = ps.calls + safeSlotsAdded + k;
      // Empirical failure probability on call >= 2 is ~41% on small budgets and scales inversely with sqrt(midBudget);
      // slight decay past call 3 accounts for surviving past the [2, budget-1] failure cluster once healthy.
      const callFactor = projectedCall <= 3 ? 1.0 : Math.max(0.75, 1.0 - 0.05 * (projectedCall - 3));
      const stepRisk = Math.round((effRiskScale * callFactor) / Math.sqrt(Math.max(2, midBudget)));
      addEdge(provNode, SINK, 1, stepRisk);
    }

    // Overflow beyond initial safe capacity carries escalating budget-exhaustion penalty.
    const overflowRisk = 4_000 + Math.round(effRiskScale);
    addEdge(provNode, SINK, m, overflowRisk);
  }

  // Push max flow via deterministic Bellman-Ford / SPFA shortest augmenting paths.
  const dist = new Array<number>(nNodes).fill(0);
  const prevNode = new Array<number>(nNodes).fill(-1);
  const prevEdge = new Array<number>(nNodes).fill(-1);
  const inQueue = new Array<boolean>(nNodes).fill(false);

  while (true) {
    dist.fill(Number.POSITIVE_INFINITY);
    prevNode.fill(-1);
    prevEdge.fill(-1);
    inQueue.fill(false);

    dist[SRC] = 0;
    const queue: number[] = [SRC];
    inQueue[SRC] = true;

    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      inQueue[u] = false;
      const edges = adj[u];
      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        if (e.cap <= 0) continue;
        const nd = dist[u] + e.cost;
        if (nd < dist[e.to]) {
          dist[e.to] = nd;
          prevNode[e.to] = u;
          prevEdge[e.to] = i;
          if (!inQueue[e.to]) {
            queue.push(e.to);
            inQueue[e.to] = true;
          }
        }
      }
    }

    if (prevNode[SINK] === -1) break;

    // Augment 1 unit (all SRC->req edges have cap 1).
    for (let v = SINK; v !== SRC; v = prevNode[v]) {
      const u = prevNode[v];
      const eIdx = prevEdge[v];
      adj[u][eIdx].cap -= 1;
      const revIdx = adj[u][eIdx].rev;
      adj[v][revIdx].cap += 1;
    }
  }

  const chosen = new Map<number, { providerIdx: number; candidate: CandidateModel }>();
  for (let r = 0; r < m; r++) {
    const reqNode = 1 + r;
    for (const ref of reqEdges[r]) {
      if (adj[reqNode][ref.edgeIdx].cap === 0) {
        chosen.set(r, { providerIdx: ref.providerIdx, candidate: ref.cand });
        break;
      }
    }
  }
  return chosen;
}

export interface RequestNeeds {
  clientWire: ClientWire;
  stream: boolean;
  tools: boolean;
  vision: boolean;
  json: boolean;
  strictJsonWire: boolean;
  minContext: number;
}

/**
 * Selects the ordered candidate to try for the current request, excluding providers already tried
 * on this request. Uses manifest lookahead when the incoming request aligns with the manifest.
 */
export function selectNextCandidate(
  state: RouterState,
  needs: RequestNeeds,
  excludedProviderIds: ReadonlySet<string>,
): CandidateModel | null {
  const manifestItem = state.cursor < state.workload.length ? state.workload[state.cursor] : null;
  const manifestMatches =
    manifestItem !== null &&
    manifestItem.api === needs.clientWire &&
    manifestItem.stream === needs.stream;

  const currentReq: Requirement = manifestMatches
    ? {
        tools: manifestItem.requires.tools === true || needs.tools,
        vision: manifestItem.requires.vision === true || needs.vision,
        json: manifestItem.requires.json === true || needs.json,
        min_context: Math.max(manifestItem.requires.min_context ?? 0, needs.minContext) || undefined,
      }
    : {
        tools: needs.tools,
        vision: needs.vision,
        json: needs.json,
        ...(needs.minContext > 0 ? { min_context: needs.minContext } : {}),
      };

  if (manifestMatches && state.workload.length > state.cursor) {
    const remaining: { requires: Requirement; weight: number; strictJsonWire: boolean }[] = [];
    for (let i = state.cursor; i < state.workload.length; i++) {
      const item = state.workload[i];
      remaining.push({
        requires: i === state.cursor ? currentReq : item.requires,
        weight: item.weight,
        strictJsonWire: i === state.cursor ? needs.strictJsonWire : item.api === 'openai' && item.requires.json === true,
      });
    }
    const marginLeft = MAX_ALLOWED_LOSS - state.totalLossSoFar;
    const assignment = solveAssignment(state.providers, remaining, excludedProviderIds, marginLeft);
    const pick = assignment.get(0);
    if (pick !== undefined) return pick.candidate;
  }

  // Fallback greedy ranking across all eligible candidates ordered by health tier then rating.
  const ranked: { ps: ProviderState; cand: CandidateModel; tier: number }[] = [];
  for (const ps of state.providers) {
    if (excludedProviderIds.has(ps.entry.id)) continue;
    const cand = bestCandidateOnProvider(ps, currentReq, needs.strictJsonWire);
    if (cand === null) continue;
    const tier = ps.consecFailures === 0 ? 0 : ps.consecFailures === 1 ? 1 : 2;
    ranked.push({ ps, cand, tier });
  }
  ranked.sort((a, b) => {
    if (a.tier !== b.tier) return a.tier - b.tier;
    if (b.cand.ratingMilli !== a.cand.ratingMilli) return b.cand.ratingMilli - a.cand.ratingMilli;
    return a.ps.entry.id < b.ps.entry.id ? -1 : a.ps.entry.id > b.ps.entry.id ? 1 : 0;
  });
  return ranked.length > 0 ? ranked[0].cand : null;
}

/** Updates provider health state and accumulated quality loss deterministically after an attempt. */
export function recordProviderOutcome(
  state: RouterState,
  providerId: string,
  ok: boolean,
  status: number,
  advanceRequestCursor: boolean,
  chosenRatingMilli?: number,
): void {
  const ps = state.providers.find((p) => p.entry.id === providerId);
  if (ps !== undefined) {
    ps.calls += 1;
    if (ok) {
      ps.successes += 1;
      ps.consecFailures = 0;
      ps.lastErrorStatus = null;
    } else {
      ps.failures += 1;
      ps.consecFailures += 1;
      ps.lastErrorStatus = status;
    }
  }
  if (advanceRequestCursor && state.cursor < state.workload.length) {
    const req = state.workload[state.cursor];
    const bestMilli = state.bestRatingByReqId.get(req.id);
    if (bestMilli !== undefined) {
      const earnedMilli = ok ? Math.min(bestMilli, chosenRatingMilli ?? bestMilli) : 0;
      state.totalLossSoFar += req.weight * (bestMilli - earnedMilli);
    }
    state.cursor += 1;
  }
}

export function getProviderWire(state: RouterState, providerId: string): Wire {
  return state.providers.find((p) => p.entry.id === providerId)?.entry.wire ?? 'openai';
}
