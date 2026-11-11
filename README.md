# Amplifier Perplexity Provider

Buffered native Perplexity Agent Responses adapter for Amplifier. The provider
identity remains `perplexity`; the module/entrypoint remains `provider-perplexity`.

> **Candidate native migration, not a published release:** This README describes
> the current local candidate. These changes have not yet been published or
> merged; a remote `@main` source must not be assumed to contain them. Offline
> SDK/HTTPX MockTransport and Core checks alone do not prove live inference, account
> entitlement, final consumer compatibility, or a regular installed-wheel smoke.

## Native API and routing

The only completion path is buffered
`await AsyncPerplexity.responses.create(...)`, serialized by SDK 0.46.1 as
`POST /v1/responses`. Perplexity's
[Agent API migration documentation](https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/how-to.md)
calls the vendor interface `/v1/agent`. This adapter does not use Sonar Chat
Completions, compatibility reformulation, or a Chat Completions fallback. Every
request sets `stream=False`, `background=False`, and `store=False`. It does not
poll, retrieve, cancel, continue a response, or dispatch client tools.

### Exact selectors

| Requested selector | Native `effort` | Native `model` |
|---|---|---|
| `agent-fast` | `fast` | omitted |
| `agent-low` | `low` | omitted |
| `agent-medium` | `medium` | omitted |
| `agent-high` | `high` | omitted |
| `sonar` | `fast` | omitted |
| `sonar-pro` | `fast` | omitted |
| `sonar-reasoning-pro` | `low` | omitted |
| `sonar-deep-research` | `high` | omitted |

The four canonical selectors and four Sonar aliases select **vendor-managed
routing**, not fixed models. Sonar aliases do not request the retired Sonar
models and do not promise equivalent behavior, price, or research depth.
Managed selectors send effort **without** a model. Any other exact nonblank ID
is passed unchanged as `model`, preserving case, **without** effort or a catalog
lookup. The service remains authoritative about that ID's availability.
Inference aliases are case-sensitive; listing filters are case-insensitive.
Blank IDs, leading/trailing padding, and unresolved globs are rejected before
hooks, client creation, or egress; an empty model does not fall back.

Vendor-managed third-party model routing, instructions, fallbacks, tools,
latency, defaults, and prices may change. The adapter explicitly adds only a
`web_search` tool, but managed profiles may merge additional search/fetch or
other tools now or later. This is not a web-only tool allowlist, local sandbox,
fixed-model contract, or deterministic research-depth guarantee. There is no
adapter client fallback chain. Prompts and tool processing leave the host for
external services, with privacy and cost implications.

`store=False` is **not** zero data retention (ZDR), deletion, or a no-retention
guarantee; the vendor may still allow continuation. Local cancellation does not
establish that server work or billing stops.

## Requirements and provider configuration

The manifest requires Python **3.11+**, registry `perplexityai>=0.46.1` and
`amplifier-core>=2.0.2`. The offline validation environment is Python 3.12.13,
SDK 0.46.1, Core 2.0.2, HTTPX 0.28.1. These are observed versions, not proof of
the entire supported dependency or Python runtime range.

### Bundle declaration versus source override

Declare the provider in a bundle. For this unpublished candidate, replace the
example URI with the absolute path of your reviewed local checkout; do not
substitute remote `@main` expecting these native changes to be there already.

```yaml
providers:
  - module: provider-perplexity
    source: file:///absolute/path/to/amplifier-module-provider-perplexity
    config:
      default_model: agent-fast
      max_tokens: 4096
      max_steps: 1
      web_search: true
```

A `~/.amplifier/settings.yaml` **module source override** is different from a
provider declaration. It changes resolution, not provider configuration:

```yaml
sources:
  modules:
    provider-perplexity: file:///absolute/path/to/amplifier-module-provider-perplexity
```

Keep generation/search settings in the bundle provider's `config`. `mount()`
uses a nonempty `config.api_key` before `PERPLEXITY_API_KEY`. Without either it
logs a warning and returns `None`, mounting no provider. Do not commit a key.
Direct `PerplexityProvider(...)` construction does not resolve the environment:
pass `api_key` explicitly for inference. Metadata and static listing can be used
without a key. Client construction is lazy; prefer `async with` or
`await provider.close()` for direct use. Mounted use returns a cleanup callback
for the host to await.

### Defaults and supported controls

Defaults: `default_model="agent-fast"`, `max_tokens=4096`, `max_steps=1`,
`web_search=True`, `timeout=None`, `max_retries=2`, `raw=False`, `debug=False`.
`temperature` and `top_p` are omitted unless resolved to a non-`None` value.
Domain/recency/type/search-token controls are omitted by default; image and
related-question flags default to `False` and are not sent.

Example provider module configuration (explicit choices, not new defaults):

```yaml
config:
  default_model: agent-low
  max_tokens: 1024
  max_steps: 1
  temperature: 0
  top_p: 1
  web_search: true
  search_domain_filter:
    - example.com
    - docs.example.com
  search_recency_filter: week
  search_type: fast
  search_max_tokens: 2048
  search_max_tokens_per_page: 512
  timeout: 60.0
  max_retries: 0
```

`max_steps` is an exact integer from 1 through 500, selected per-call before
config before default 1. It is not a bound on tool count, dollars, or elapsed
time. `temperature` must be a finite number in 0..2; `top_p` in 0..1. Booleans
are not generation numbers; numeric zero is preserved.

### Per-call precedence and budgets

`complete(request, **kwargs)` accepts supported controls directly or through a
`request_options` mapping (or `None`). Direct keywords replace mapped values by
**key presence**, retaining explicit `None`, `False`, and `0`. Inputs are copied
for conversion/wire/event snapshots; there is no arbitrary SDK passthrough.

After that merge, generation values use the first non-`None` value:

| Setting | Selection order |
|---|---|
| model | call `model` > `ChatRequest.model` > config `default_model` (default `agent-fast`) |
| output budget | call legacy `max_tokens` > `ChatRequest.max_output_tokens` > config `max_tokens` > 4096 |
| temperature / top_p | call > corresponding Core field > config > `None` |

For example, direct `max_tokens=None` shadows a mapped `max_tokens=0`, then
falls through to the Core/config budget; direct `0` does not fall through.
**An effective output budget must be an exact `int >= 1`.** Zero, negative
values, and booleans raise `ValueError` before egress; zero is no longer forwarded
for remote validation. Catalog limits of zero mean unknown metadata and are
categorically different from generation budgets.

Core requests use `ChatRequest.max_output_tokens`. A per-call
`max_output_tokens=` keyword (including one inside `request_options`) is
rejected with guidance to use that Core field or legacy `max_tokens=`; it is
not silently ignored. `ChatRequest.timeout` is not adopted: use the explicit
timeout paths below. Conversation IDs and Core metadata remain host state.

### Native web search and migration from retired options

Search controls and timeout select **direct keyword > mapping > config by
presence**, not truthiness. Omission inherits; explicit clears replace config.

| Control | Accepted value / clear behavior |
|---|---|
| `web_search` | Actual bool; default `True`; restricted disabling described below |
| `search_domain_filter` | `list[str]`, at most 20 nonblank entries; `None` / `[]` clears |
| `search_recency_filter` | `hour`, `day`, `week`, `month`, `year`; `None` / `"none"` clears |
| `search_type` | `fast`, `web`; `None` clears; config-only UI `"inherit"` means omission |
| `search_max_tokens` | Exact positive integer; `None` clears |
| `search_max_tokens_per_page` | Exact positive integer; `None` clears |

Domain lists must be **all inclusions** or **all `-` exclusions**, never mixed.
Call-level `search_type="inherit"` is invalid; that spelling is only normalized
from config. Native search parameters go inside `tools`: token controls become
`max_tokens` / `max_tokens_per_page`, type becomes `search_type`, and domains /
recency go under the tool's `filters`. They are not Sonar top-level fields or
`web_search_options`. The YAML above produces a tool shaped like this (other
request fields omitted here):

```json
{
  "type": "web_search",
  "filters": {
    "search_domain_filter": ["example.com", "docs.example.com"],
    "search_recency_filter": "week"
  },
  "search_type": "fast",
  "max_tokens": 2048,
  "max_tokens_per_page": 512
}
```

Migration rules:

- Non-`None` `search_context_size` is unsupported. Clear with `None` and use
  `search_max_tokens` / `search_max_tokens_per_page`; no invented low/medium/high
  token conversion is provided.
- `return_images` / `return_related_questions` require actual booleans.
  `False` clears/omits; `True` is rejected before egress. No automatic paid
  image-search tool or follow-up-question prompt is substituted.
- Any non-`None` `reasoning_effort`, including `"none"`, is rejected. Managed
  selector effort is independent. A `None` keyword cannot mask an explicit
  non-`None` Core `reasoning_effort`.
- `web_search=False` is allowed only with an exact direct native model ID and
  all active search controls cleared. It omits `tools` entirely. Managed
  selectors and Sonar aliases cannot disable search: sending `tools=[]` cannot
  turn off vendor effort tools and is not an adapter override. Clearing adapter
  filters does not erase server profile defaults or guarantee no vendor tools.

Typed request example with supported fields (makes a live request if called):

```python
from amplifier_core.message_models import ChatRequest, ChatResponse, Message
from amplifier_module_provider_perplexity import PerplexityProvider


async def search_example(provider: PerplexityProvider) -> ChatResponse:
    request = ChatRequest(
        model="agent-low",
        messages=[Message(role="user", content="Summarize recent public research.")],
        max_output_tokens=512,
        temperature=0,
        top_p=1,
    )
    return await provider.complete(
        request,
        request_options={"search_recency_filter": "week", "search_max_tokens": 2048},
        search_domain_filter=["example.com"],
        search_max_tokens_per_page=512,
        max_steps=1,
    )
```

### Unsupported request intent fails before egress

The adapter accepts nonempty text-only history with `system`, `developer`,
`user`, and `assistant` roles, preserving roles and order without prompt
wrapping. Lists of Core `TextBlock`s are joined with newlines. Nonempty client
tools, tool history/tool-result roles, non-`None` tool choice, media/reasoning
blocks, JSON/schema response modes, nonempty stop sequences, and streaming
requests are rejected, not silently dropped. `parse_tool_calls()` returns `[]`;
server-side vendor tools are not client tool calls.

Explicit native override keywords `extra_body`, `extra_headers`, `extra_query`,
`effort`, `profile`, `preset`, `models`, `previous_response_id`, `instructions`,
`tools`, `skills`, `tool_choice`, `store`, and `background` are unsupported even
when supplied as clears. `stream=True` is unsupported. Unrelated host bookkeeping
is not forwarded to the SDK. There is no new tool client, hook-approval protocol,
or automatic continuation path.

## Static model recommendations, not capacity admission

`await provider.list_models()` returns the four canonical `agent-*` rows by
default, in the selector table's order. `config.model_filters` is frozen at
construction. `filters=None` uses that snapshot; an explicit mapping **replaces**
it, never merges. `filters={}` returns **all eight** rows in table order. Config
`include=["sonar*"]` selects the four alias rows, not a vendor model inventory.

```python
from amplifier_module_provider_perplexity import PerplexityProvider


async def listing_example(provider: PerplexityProvider) -> dict[str, list[str]]:
    default_rows = await provider.list_models(filters=None)
    all_rows = await provider.list_models(filters={})
    aliases = await provider.list_models(filters={"include": ["SONAR*"]})
    empty = await provider.list_models(filters={"include": []})
    return {
        "configured": [row.id for row in default_rows],
        "all_eight": [row.id for row in all_rows],
        "aliases": [row.id for row in aliases],
        "empty": [row.id for row in empty],
    }
```

Include/exclude use case-insensitive OS-independent globs; exclusion wins.
`include=[]` and `exclude=["*"]` select nothing. `hide_dated=True` suppresses
`-YYYY-MM-DD` snapshots only when the corresponding undated ID is selected.
Invalid filter keys/values are rejected. Descriptors, schemas, returned models,
and collections are fresh snapshots, not mutable policy aliases. Exact inference
is independent of catalog visibility, even with an empty listing.

`get_model_list_info()` retains the optional v1 contract: `version=1`,
`source="static"`, `filters_keyword="filters"`, `filters_schema` from
`selector_schema()`, frozen `default_filters`, `unfiltered={}`, and
`override="replace"`. Listing and metadata construct no SDK client, require no
authenticated enumeration or Foundation dependency, and establish no account
entitlement or admission gate.

Every row has `context_window=0`, `max_output_tokens=0`, `defaults={}`, and
`pricing=None`. **Numeric zero here means UNKNOWN**, not zero usable capacity,
a request cap, a generation default, or a validated context/output limit. No
historical Sonar capacities or fixed prices are asserted for native routing.
Provider/model capabilities are only `web_search` and `citations`: adapter
surfaces, not guaranteed retrieval/citation output. No streaming, client tools,
image input, thinking blocks, or fixed research-depth support is advertised.

## Direct usage and cleanup

This example requires a real key and service access **if executed**. Documentation
validation compiles examples only; it does not execute them or claim live success.

```python
import asyncio
import os

from amplifier_core.message_models import ChatRequest, Message, TextBlock
from amplifier_module_provider_perplexity import PerplexityProvider


async def main() -> None:
    request = ChatRequest(
        model="agent-fast",
        messages=[Message(role="user", content="Summarize recent quantum research.")],
        max_output_tokens=256,
    )
    async with PerplexityProvider(
        api_key=os.environ["PERPLEXITY_API_KEY"],
        config={"max_retries": 0},
    ) as provider:
        response = await provider.complete(request, search_recency_filter="week")
    for block in response.content:
        if isinstance(block, TextBlock):
            print(block.text)
    metadata = response.metadata or {}
    print("Requested selector:", metadata.get("perplexity:requested_selector"))
    print("Actual serving model:", metadata.get("perplexity:model"))
    print("Cited URLs:", metadata.get("perplexity:citations", []))


if __name__ == "__main__":
    asyncio.run(main())
```

## Response text, native metadata, and usage

Core `ChatResponse` has **no `model` field**. The requested selector is separate
from the actual returned model; missing/blank returned model becomes `None`,
never a fallback to the requested alias.

| Namespaced metadata key | Meaning |
|---|---|
| `perplexity:response_id` | Native response identity |
| `perplexity:requested_selector` | Caller/default selector actually resolved |
| `perplexity:resolved_effort` | Adapter-resolved managed effort, when applicable |
| `perplexity:model` | Actual returned serving-model string or `None` |
| `perplexity:effort` | Returned native effort, when present |
| `perplexity:status` | Returned accepted terminal status |
| `perplexity:incomplete_details` | Selected object `{"reason":"<returned reason>"}` when the returned reason is a string |
| `perplexity:search_results` | Flattened native results with native IDs and supported source fields |
| `perplexity:search_result_items` | Per-output envelopes: output index, results, returned queries/IDs/correlation fields |
| `perplexity:annotations` | Returned annotations grouped by output/content index |
| `perplexity:citations` | URLs from returned `url_citation` annotations only |
| `perplexity:usage_cost` | Sanitized native JSON cost fields, separately retained when returned |
| `perplexity:usage_cost_status` | `known`, `missing`, `invalid`, or `non_usd` |

Only assistant `output_text` parts become answer text, concatenated in order
into a Core `TextBlock`. No synthetic numbered Sources list is appended. Reasoning,
server-tool traces, arbitrary media text, and service error prose are not rendered
as answers. There are no image/related-question metadata extras.

Retrieved URLs are **not** cited URLs. Optional metadata is included when its
supported native field is returned; explicit empty arrays survive. Inline citation
markers can occur with empty annotations. Resolve native source IDs using returned
results/envelopes, not newly assigned flattened-list positions; do not invent
citations from retrieved URLs. Treat returned text, URLs, and annotations as
untrusted external data.

`completed` maps to `finish_reason="stop"`. `incomplete` maps to `"length"`
**only** for returned `incomplete_details.reason="max_output_tokens"`; other
incomplete reasons yield `None` and retain the returned reason metadata. Token
counts do not imply truncation. SDK 0.46.1's strict schema lacks the `incomplete`
status: offline proof of permissive HTTP decoding of that drift is explicitly
distinct from strict-valid completed fixture proof, not live service validation.
The metadata value is the selected reason object, not a string-valued
`incomplete_details` or a copy of the entire vendor object.

Missing/null usage becomes `None`. Native input/output/**total** token counts
are preserved; total is not invented by summing. Optional cache-read/cache-write
and reasoning counts are populated only from returned fields, including zeros.
A finite nonnegative USD `total_cost` becomes Core `Usage.cost_usd` using
`Decimal(str(SDK_value))`, not lossless recovery of original wire decimal spelling.
Missing/invalid/non-USD costs yield `cost_usd=None` and the corresponding explicit
cost status, **never an invented $0/free result**. Sanitized native cost metadata
is JSON-safe; the Decimal lives in Core Usage memory and serializes in a Core JSON
dump.

## Transport waits, cancellation, errors, and observations

Healthy generation has no default read/write or total elapsed deadline. A scalar
`timeout` or `None` becomes `perplexity.Timeout(value, connect=5, pool=5)`; zero
is preserved. Default read/write waits are `None`. Structured `perplexity.Timeout`
preserves all four explicit `connect`, `read`, `write`, and `pool` phases,
including `None` and zero, **without** imposing five-second bounds.

Timeout can be configured on construction, in `request_options`, or directly;
direct > mapping > config by presence. Explicit `timeout=None` clears configured
read/write waits while retaining scalar-form connect/pool bounds. Overrides do
not mutate the cached client or caller inputs. These are **phase timeouts**, not
a total elapsed deadline; slow generation alone is not a defect.

Use practical asyncio task cancellation: `CancelledError` propagates. Core
CancellationToken arguments/coordinator tokens are not polled; hooks are not
token gates. No watchdog, hidden task, shield, server cancellation request, or
billing-stop guarantee is implemented. A server `cancelled` response is an error
state, not a local task's cancellation signal.

SDK retries are authoritative (default two); there is no second provider retry
loop or automatic output-budget escalation. The SDK retries connection errors
and HTTP 408/409/429/5xx with backoff; configure `max_retries` at construction.
Native HTTP/transport exceptions retain object, body, and cause identity when
propagated. Provider-owned failure logs record only exception class/status, not
body/message/traceback/cause. Native exceptions can contain sensitive vendor
content: the caller/host must handle them accordingly.

Preflight `ValueError`s are non-echoing. Failed/cancelled/nonterminal/malformed
HTTP-200 native responses produce fixed safe `LLMError`s with provider
`perplexity`, status 200, `retryable=False`, and no vendor body or chained cause.
No `llm:response` success event is emitted for those responses, and there is no
automatic continuation. `llm:request` and `llm:response` are same-task observations,
not approval/denial controls. There is no new `provider:error` emitter or change
to host event ownership.

`raw=True` opts into complete converted native prompt/request capture on
`llm:request`, with **key-based** redaction only. Full prompts, PII, and secrets
embedded in text remain disclosed. The captured snapshot cannot mutate wire
parameters. Explicit timeouts appear as JSON-safe phase dictionaries, scalars,
or `None`; inherited config timeout is not injected as a per-call raw override.
Third-party SDK/HTTPX DEBUG logging is outside the provider-owned logging guarantee
and can expose content independently of `raw=False`. Configure host logging and
event storage accordingly.

## Development and verification boundaries

The dev dependency group includes `httpx>=0.28.1`. With tools already installed,
the repository's standard offline checks are:

```bash
ruff format --check .
ruff check .
pyright
python -m pytest tests
```

For credential-free restricted verification, run these through the qualified
offline guard, disable bytecode/test/lint caches and plugin autoload, and invoke
the **original installed Node Pyright entrypoint**, not an updater launcher.
A Python 3.11 static target check is not Python 3.11 runtime qualification.
Compile every Python fence independently without executing examples.

The suite uses HTTPX MockTransport for native SDK dispatch; the commands above
do not make live service calls. Offline Core structural/behavioral admission
checks are supplemented by typed native-wire completion/hooks/cleanup tests;
the inherited checks do not exercise `complete()`. These checks do not prove
remote model availability, entitlement, prices, capacities, all Python versions,
or dependency ranges.

Verify actual import paths and file hashes separately for source and regular-wheel
runs: an older installed wheel does not qualify changed checkout bytes. Report
final-byte wheel/source joins, pinned actual-consumer qualification, reviews, and
bounded live source/packaged evidence separately in the PR. Consumer prerequisite
results from an older provider do not qualify a migrated candidate. Live evidence
is specific to the tested bytes, account, selectors, and time; it is not a permanent
availability or entitlement guarantee. This README does not attest publication,
merge, or future live success.

## License

MIT
