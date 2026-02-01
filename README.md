# Amplifier Perplexity Provider

Perplexity AI provider for Amplifier, enabling web-grounded AI search with real-time citations.

## Installation

```bash
# Using uv
uv add amplifier-module-provider-perplexity

# Using pip
pip install amplifier-module-provider-perplexity
```

## Configuration

### Environment Variables

Set your Perplexity API key:

```bash
export PERPLEXITY_API_KEY="pplx-xxxxxxxxxxxxxxxx"
```

### settings.yaml

Add the provider to your Amplifier configuration:

```yaml
modules:
  - name: provider-perplexity
    config:
      # Optional: Override default model
      default_model: sonar-pro
      
      # Optional: Search settings
      search_recency_filter: week  # none, hour, day, week, month, year
      search_context_size: medium  # low, medium, high
      search_domain_filter:
        - example.com
        - docs.example.com
      
      # Optional: Additional response data
      return_images: false
      return_related_questions: false
      
      # Optional: Request settings
      max_tokens: 4096
      temperature: 0.7
      timeout: 60.0
      max_retries: 3
```

## Available Models

| Model | Context Window | Max Output | Capabilities |
|-------|---------------|------------|--------------|
| `sonar` | 128k | 4,096 | Web search, citations, streaming |
| `sonar-pro` | 200k | 8,192 | Web search, citations, streaming, deep search |
| `sonar-reasoning` | 128k | 4,096 | Web search, citations, streaming, reasoning |
| `sonar-reasoning-pro` | 128k | 8,192 | Web search, citations, streaming, reasoning, deep search |
| `sonar-deep-research` | 128k | 8,192 | Web search, citations, deep research |
| `r1-1776` | 128k | 4,096 | Offline (no web search) |

## Perplexity-Specific Features

### Citations

Perplexity responses include citations to web sources. Access them via response metadata:

```python
response = await provider.complete(request)
citations = response.metadata.get("citations", [])
for citation in citations:
    print(f"Source: {citation['url']}")
```

### Search Recency Filter

Control how recent search results should be:

- `none` - No filter (default)
- `hour` - Last hour
- `day` - Last 24 hours
- `week` - Last 7 days
- `month` - Last 30 days
- `year` - Last year

### Search Context Size

Control how much search context is included:

- `low` - Minimal context, faster responses
- `medium` - Balanced (default)
- `high` - Maximum context, more comprehensive

### Domain Filtering

Restrict searches to specific domains:

```yaml
config:
  search_domain_filter:
    - wikipedia.org
    - arxiv.org
```

### Related Questions

Enable related question suggestions:

```yaml
config:
  return_related_questions: true
```

Access via:

```python
related = response.metadata.get("related_questions", [])
```

## Usage Example

```python
from amplifier_core.message_models import ChatRequest, Message

# Create a request
request = ChatRequest(
    model="sonar",
    messages=[
        Message(role="user", content="What are the latest developments in quantum computing?")
    ],
    max_tokens=2000,
)

# Get response with citations
response = await provider.complete(request)

print(response.content[0].text)

# Access citations
if response.metadata and "citations" in response.metadata:
    print("\nSources:")
    for citation in response.metadata["citations"]:
        print(f"  - {citation.get('title', 'Untitled')}: {citation['url']}")
```

## Error Handling

The provider implements automatic retry with exponential backoff for:

- **429 (Rate Limited)**: Uses `retry-after` header when available
- **5xx (Server Errors)**: Retries with exponential backoff

Errors that are not retried:

- **401 (Unauthorized)**: Invalid API key - fails immediately

## License

MIT
