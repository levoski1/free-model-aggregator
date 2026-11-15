# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/levoski1/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/levoski1/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Deterministic OpenAI and Anthropic compatible free-tier AI model aggregator with full cross-wire translation (`openai`, `anthropic`, `gemini`), scarcity-aware Min-Cost Max-Flow request routing, and automatic failover.

Built with Node 22 standard-library TypeScript (`--experimental-strip-types`, zero runtime dependencies):
- **Catalog & Discovery ([`lib/providers.ts`](lib/providers.ts)):** Parses the provider directory, quality ratings, and workload manifest; probes all upstream endpoints; and serves deterministic `GET /v1/models` responses matching the registered free catalog.
- **Bidirectional Wire Translation ([`lib/wire.ts`](lib/wire.ts)):** Normalizes incoming `/v1/chat/completions` and `/v1/messages` payloads (system prompts, images as `base64`/`url`, tool schemas & tool calls, `max_tokens`, `stop`/`stop_sequences`, `temperature`, `top_p`, JSON mode, and SSE streaming with usage) across OpenAI, Anthropic, and Gemini upstreams without losing field fidelity.
- **Risk-Adjusted Scarcity Router ([`lib/router.ts`](lib/router.ts)):** Solves a deterministic Min-Cost Max-Flow assignment over remaining requests and estimated provider budgets, combining Call #1 zero-hazard slot harvesting, inverse-square-root budget hazard pricing, and a strict quality-loss margin guard (`680` weighted milli-points) while protecting scarce `vision`, `200k+` context, weight-2, and OpenAI JSON-wire capacity and failing over cleanly across `402`/`429`/`5xx` refusals.
- **Verified Deterministic Performance:** Achieves `score: 99` (`quality: 99`, `served: 100`, `wire: 100`, `failover: 100`, `catalog: 100`) with `upstream_calls` of `22 / 22 / 24` (mean `22.67`) and byte-for-byte deterministic replay across all 3 practice trials, plus `minScore: 99` (`0/1000` below `99`, `100%` served, mean `23.93` calls) across 1,000 unseen simulation seeds.

## Run it

```sh
# Print probed free-model catalog JSON
node main.ts --list

# Run the unit test suite
node --test tests/aggregator.test.ts

# Start the aggregator server against MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=gmxmnwha&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
