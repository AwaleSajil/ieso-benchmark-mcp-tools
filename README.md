# IESO Benchmark Tools

FastMCP server exposing two MCP tools for the **IESO agent synthetic benchmark generation pipeline**. The tools are called by agents built in OpenAI Agent Builder to map satellite dataset metadata (extracted from scientific papers) to NASA Worldview layers and validate their temporal coverage.

---

## Context

The IESO agent generates NASA Worldview deep links from natural-language queries. To benchmark it, we generate synthetic query–answer pairs from peer-reviewed papers that use satellite data. These MCP tools handle the **System Validation** step of that pipeline — ensuring every benchmark query maps to a real, temporally valid Worldview layer.

```
PDF Paper
    │
    ▼
Extraction Agent (OpenAI Agent Builder)
    │  outputs datasets[] with satellite metadata
    ▼
Benchmark Agent (OpenAI Agent Builder)
    ├── map_metadata_to_worldview_layers(dataset)  ←── this server
    │       returns matched layer IDs + scores
    ├── validate_temporal_coverage(layer_id, acquisition_start_date, acquisition_end_date) ←── this server
    │       returns complete/partial/no overlap for the requested acquisition window
    └── generates benchmark query (no dataset names, natural language)
```

---

## Tools

### `map_metadata_to_worldview_layers`

Maps one dataset object (from the extraction agent) to NASA Worldview layer IDs using hybrid search (vector + BM25) over the local layer catalog.

**Input** — one item from the extraction agent's `datasets` array:

```json
{
  "Satellite data name":    "MODIS Terra",
  "Sensor Name":            "MODIS",
  "Variable / Measurement": "land surface temperature",
  "Phenomenon":             "urban heat island",
  "Science topic":          "atmospheric science",
  "Spatial Resolution":     "1km",
  "Temporal Resolution":    "daily",
  "Acquisition Start Date": "2019",
  "Acquisition End Date":   "2021",
  "Location Coverage":      "Indian subcontinent",
  "Processing Level":       "L3",
  "How used":               "to quantify surface warming in cities"
}
```

**Output:**

```json
{
  "matched_layers": [
    {
      "layer_id": "MODIS_Terra_Land_Surface_Temp_Day",
      "display_name": "...",
      "description": "...",
      "date_range_start": "2000-02-24",
      "date_range_end": null,
      "instrument": "modis",
      "platform": "Terra",
      "tags": ["lst", "land surface temperature"],
      "relevance_score": 0.91,
      "vector_score": 0.88,
      "bm25_score": 0.76,
      "search_mode": "hybrid"
    }
  ],
  "query_used": "land surface temperature urban heat island MODIS Terra ...",
  "acquisition_dates_parsed": ["2019-07-15", "2020-07-15", "2021-07-15"],
  "dataset_echo": { ... },
  "total_layers_in_catalog": 1218
}
```

---

### `validate_temporal_coverage`

Checks overlap between a requested acquisition window and a specific Worldview layer's temporal coverage.

**Input:**

```json
{
  "layer_id": "MODIS_Terra_Land_Surface_Temp_Day",
  "acquisition_start_date": "2019",
  "acquisition_end_date": "2021"
}
```

**Output:**

```json
{
  "layer_id": "MODIS_Terra_Land_Surface_Temp_Day",
  "layer_start": "2000-02-24",
  "layer_end": null,
  "is_currently_active": true,
  "requested_window": { "start": "2019-07-15", "end": "2021-07-15" },
  "overlap_type": "COMPLETE_OVERLAP",
  "overlap_window": { "start": "2019-07-15", "end": "2021-07-15" },
  "not_covered_windows": [],
  "overall_valid": true,
  "source": "json"
}
```

Possible `overlap_type` values:

| Type | Meaning |
|---|---|
| `COMPLETE_OVERLAP` | Requested acquisition window is fully covered by layer availability |
| `PARTIAL_OVERLAP` | Only part of the requested window is covered |
| `NO_OVERLAP` | Requested window does not overlap layer availability |

---

## Search architecture

Queries arrive as plain text. The tool embeds them at call time and runs hybrid search locally — no external vector database required.

```
Tool receives plain-text dataset fields
    │
    ├── Build query string from fields
    │
    ├── Embed query → OpenAI text-embedding-3-small (1 API call, ~50ms)
    │
    ├── Cosine similarity vs 1 218 pre-computed layer embeddings (numpy, <1ms)
    │
    ├── BM25 keyword score (rank-bm25, <1ms)
    │
    ├── Hybrid score = 0.7 × vector + 0.3 × BM25
    │
    └── Post-filter by Acquisition Start/End Date range → return top-k
```

Both indexes (BM25 + numpy embedding matrix) are built once at server startup and kept in memory for the lifetime of the process.

---

## Files

```
ieso_benchmark_tools/
├── server.py                    ← FastMCP server (the two tools)
├── generate_embeddings.py       ← one-time script to add embeddings to the JSON
├── test_local.py                ← local integration tests
├── worldview_unified_final.json ← 1 218 Worldview layers (add embeddings before deploy)
├── pyproject.toml               ← uv project config
├── .env.example                 ← environment variable template
└── README.md                    ← this file
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set:
#   OPENAI_API_KEY=sk-...
```

### 3. Generate layer embeddings (one-time)

Adds a `"embedding"` vector to each layer in `worldview_unified_final.json`.
Safe to re-run — skips layers that already have embeddings.

```bash
uv run python generate_embeddings.py
```

Cost: ~$0.005. Time: ~10 seconds for 1 218 layers.

Without embeddings the server still works but falls back to BM25-only search
(`search_mode: "bm25_only"` in results).

### 4. Test locally

```bash
uv run python test_local.py
```

Or open the interactive MCP inspector in a browser:

```bash
uv run fastmcp dev inspector server.py:mcp
# → http://localhost:6274
```

---

## Deploy to FastMCP Cloud

FastMCP CLI behavior differs by version. In newer versions, `fastmcp deploy`
may not be available as a direct command.

1. Check your CLI:

```bash
uv run fastmcp --help
```

2. If your version supports `deploy`, use:

```bash
uv run fastmcp deploy server.py
```

3. If `deploy` is not listed, deploy through the FastMCP Cloud UI/workflow for
your account, then use the resulting server URL (for example:
`https://your-server.fastmcp.cloud/mcp`).

After deployment, set `OPENAI_API_KEY` in FastMCP Cloud environment variables
so query embedding works at runtime.

---

## Connect to OpenAI Agent Builder

In the Agent Builder UI, add an MCP server:

```
URL:   https://your-server.fastmcp.cloud/mcp
Label: ieso-benchmark-tools
```

The agent builder auto-discovers both tools from the server schema. No manual tool definition needed.

Or via the OpenAI Responses API:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_url": "https://your-server.fastmcp.cloud/mcp",
        "server_label": "ieso-benchmark-tools",
    }],
    input="Here is a paper about wildfire monitoring using MODIS Terra data..."
)
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Used to embed incoming queries. |
| `WORLDVIEW_JSON_PATH` | `worldview_unified_final.json` | Path to the layer catalog JSON. |
| `HYBRID_ALPHA` | `0.7` | Weight of vector vs BM25 score. `1.0` = pure vector, `0.0` = pure BM25. |
