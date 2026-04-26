"""
IESO Benchmark Tools – FastMCP Server

Two MCP tools for the synthetic benchmark generation pipeline.  Fully
self-contained: no Weaviate.  The layer catalog is loaded from a local JSON
file (worldview_unified_final.json).  Pre-computed embeddings are loaded from
a companion binary file (<stem>_embeddings.npy) created by generate_embeddings.py.
Hybrid search (cosine-similarity + BM25) is executed in-process with numpy + rank_bm25.

Search pipeline (map_metadata_to_worldview_layers):
  1. Build a text query from the extraction-agent dataset fields.
  2. Embed the query with OpenAI text-embedding-3-small.
  3. Cosine-similarity against all pre-computed layer embeddings → vector_score.
  4. BM25 keyword score against layer text fields → bm25_score.
  5. Hybrid score = ALPHA * vector_score + (1 - ALPHA) * bm25_score.
  6. Post-filter by date range (Acquisition Start/End Date from the paper).
  7. Return top-k layers.

If the embedding file is absent, the tool falls back to BM25-only and logs a warning.
Run generate_embeddings.py once to create the .npy file.

Extraction agent dataset schema (one item from the "datasets" array):
  {
    "Satellite data name":    str,
    "Sensor Name":            str,
    "Variable / Measurement": str,
    "Phenomenon":             str,
    "Science topic":          str,
    "Spatial Resolution":     str,
    "Temporal Resolution":    str,
    "Acquisition Start Date": str,   # YYYY, YYYY-MM, or YYYY-MM-DD
    "Acquisition End Date":   str,   # YYYY, YYYY-MM, or YYYY-MM-DD
    "Location Coverage":      str,
    "Processing Level":       str,
    "How used":               str,
  }

Environment variables (see .env.example):
  OPENAI_API_KEY      Required for query embedding (text-embedding-3-small).
  WORLDVIEW_JSON_PATH Path to worldview_unified_final.json.
                      Defaults to worldview_unified_final.json in the same
                      directory as this file.
                      Companion embedding files (<stem>_embeddings.npy and
                      <stem>_ids.json) are expected in the same directory.
  HYBRID_ALPHA        Float 0–1, weight of vector vs BM25 (default 0.7).
                      1.0 = pure vector, 0.0 = pure BM25.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "ieso-benchmark-tools",
    instructions=(
        "Tools that support synthetic benchmark generation for the IESO agent. "
        "Use map_metadata_to_worldview_layers to find Worldview layers that match "
        "metadata extracted from a scientific paper. "
        "Use validate_temporal_coverage to confirm a layer was active during the "
        "paper's acquisition dates before finalising a benchmark query."
    ),
)

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_JSON = os.path.join(
    os.path.dirname(__file__), "worldview_unified_final.json"
)
WORLDVIEW_JSON_PATH = os.getenv("WORLDVIEW_JSON_PATH", _DEFAULT_JSON)
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))   # vector weight
EMBED_MODEL  = "text-embedding-3-small"

# Companion embedding archive derived from JSON path
_json_stem      = os.path.splitext(WORLDVIEW_JSON_PATH)[0]
EMBEDDINGS_PATH = _json_stem + "_embeddings.npz"

# ── In-memory index (lazy, loaded once) ──────────────────────────────────────

_layers:      list[dict[str, Any]] = []   # full layer dicts (embedding stripped)
_layer_id_map: dict[str, dict[str, Any]] = {}  # layer_id → layer dict  (O(1) lookup)
_bm25:        BM25Okapi | None = None
_embeddings:  np.ndarray | None = None    # shape (N, D), float32; None = not available
_index_ready  = False


def _layer_text(layer: dict[str, Any]) -> str:
    """Concatenate the text fields used for BM25 and embedding."""
    fields = [
        ("layer_id",    layer.get("layer_id", "")),
        ("name",        layer.get("display_name", "")),
        ("description", layer.get("description", "")),
        ("tags",        " ".join(layer.get("tags", []))),
        ("instrument",  layer.get("instrument", "")),
        ("platform",    layer.get("platform", "")),
        ("measurement", layer.get("measurement_id", "")),
    ]
    return " ".join(f"{k}: {v}" for k, v in fields if v).lower()


def _build_index() -> None:
    global _layers, _layer_id_map, _bm25, _embeddings, _index_ready

    if _index_ready:
        return

    if not os.path.exists(WORLDVIEW_JSON_PATH):
        raise FileNotFoundError(
            f"Layer catalog not found: {WORLDVIEW_JSON_PATH}\n"
            f"Set WORLDVIEW_JSON_PATH or place worldview_unified_final.json "
            f"next to server.py."
        )

    logger.info(f"Loading layer catalog from {WORLDVIEW_JSON_PATH} …")
    with open(WORLDVIEW_JSON_PATH) as f:
        data = json.load(f)

    raw_layers: list[dict[str, Any]] = data.get("layers", [])
    logger.info(f"  {len(raw_layers)} layers found")

    for layer in raw_layers:
        layer.pop("embedding", None)   # strip legacy field if still present
        _layers.append(layer)
        _layer_id_map[layer.get("layer_id", "")] = layer

    # Build BM25 index over the text of every layer
    tokenised = [_layer_text(l).split() for l in _layers]
    _bm25 = BM25Okapi(tokenised)
    logger.info("  BM25 index built")

    # Load embeddings from companion .npz archive (ids and embeddings bundled together)
    if os.path.exists(EMBEDDINGS_PATH):
        archive = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        emb_ids: list[str] = [str(x) for x in archive["ids"]]
        matrix = archive["embeddings"]
        id_to_row = {lid: i for i, lid in enumerate(emb_ids)}

        # Align embedding rows to the layer order in the JSON
        aligned: list[np.ndarray] = []
        missing = 0
        for layer in _layers:
            row = id_to_row.get(layer.get("layer_id", ""))
            if row is not None:
                aligned.append(matrix[row])
            else:
                missing += 1

        if missing == 0 and aligned:
            _embeddings = np.array(aligned, dtype=np.float32)
            norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _embeddings /= norms
            logger.info(f"  Embedding matrix: {_embeddings.shape}")
        else:
            logger.warning(
                f"  {missing} layer(s) missing from embedding file — "
                "vector search disabled, using BM25-only."
            )
    else:
        logger.warning(
            f"  {os.path.basename(EMBEDDINGS_PATH)} not found — "
            "using BM25-only search. Run generate_embeddings.py to create it."
        )

    _index_ready = True


# ── OpenAI embedding ──────────────────────────────────────────────────────────

_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI | None:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY not set — vector search disabled")
            return None
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _embed_query(query: str) -> np.ndarray | None:
    """Return a unit-norm embedding vector for the query, or None on failure."""
    client = _get_openai()
    if client is None or _embeddings is None:
        return None
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=query)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


# ── Hybrid search ─────────────────────────────────────────────────────────────

def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if all scores are equal."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _hybrid_search(
    query: str,
    norm_dates: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Run hybrid (vector + BM25) search over the in-memory index.

    Steps:
      1. BM25 keyword scores over all layers.
      2. Cosine-similarity vector scores (if embeddings available).
      3. Combine: ALPHA * vector + (1 - ALPHA) * BM25.
      4. Sort descending, post-filter by date range, return top-k.
    """
    _build_index()
    n = len(_layers)

    # ── BM25 scores ───────────────────────────────────────────────────────────
    tokens = query.lower().split()
    raw_bm25 = np.array(_bm25.get_scores(tokens), dtype=np.float32)
    bm25_norm = _normalise(raw_bm25)

    # ── Vector scores (cosine similarity) ─────────────────────────────────────
    q_vec = _embed_query(query) if _embeddings is not None else None
    if q_vec is not None and _embeddings is not None:
        # Dot product with unit-norm rows = cosine similarity
        vec_scores = (_embeddings @ q_vec).astype(np.float32)
        vec_norm = _normalise(vec_scores)
        alpha = HYBRID_ALPHA
    else:
        vec_norm = np.zeros(n, dtype=np.float32)
        alpha = 0.0   # fall back to pure BM25

    hybrid = alpha * vec_norm + (1.0 - alpha) * bm25_norm

    # ── Sort and filter ───────────────────────────────────────────────────────
    # Over-fetch to allow for date filtering downstream
    top_k = min(limit * 6, n)
    top_indices = np.argsort(hybrid)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for idx in top_indices:
        layer = _layers[idx]
        l_start_str = layer.get("date_range_start")
        l_end_str   = layer.get("date_range_end")

        # Date range filter
        if norm_dates and l_start_str:
            l_start_dt = _parse_date(l_start_str)
            l_end_dt   = _parse_date(l_end_str)
            if not any(
                _date_in_range(_parse_date(d), l_start_dt, l_end_dt)
                for d in norm_dates
            ):
                continue

        results.append({
            "layer_id":        layer.get("layer_id"),
            "display_name":    layer.get("display_name"),
            "description":     (layer.get("description") or "")[:400],
            "date_range_start": l_start_str,
            "date_range_end":   l_end_str,
            "instrument":      layer.get("instrument"),
            "platform":        layer.get("platform"),
            "tags":            layer.get("tags", []),
            "relevance_score": round(float(hybrid[idx]), 4),
            "vector_score":    round(float(vec_norm[idx]), 4),
            "bm25_score":      round(float(bm25_norm[idx]), 4),
            "search_mode":     "hybrid" if alpha > 0 else "bm25_only",
        })

        if len(results) >= limit:
            break

    return results


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    raw = str(s).strip()
    for fmt, length in [
        ("%Y-%m-%dT%H:%M:%SZ", 20),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d", 10),
    ]:
        try:
            return datetime.strptime(raw[:length], fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _normalise_dates(raw: list[str]) -> list[str]:
    """Coerce YYYY, YYYY-MM, YYYY-MM-DD to YYYY-MM-DD."""
    out: list[str] = []
    for d in raw:
        s = str(d).strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            out.append(s[:10])
        elif len(s) == 7 and s[4] == "-":
            out.append(f"{s}-15")
        elif len(s) == 4 and s.isdigit():
            out.append(f"{s}-07-15")
    return out


def _date_in_range(
    dt: datetime | None,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if dt is None:
        return False
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


# ── Tool 1: map_metadata_to_worldview_layers ──────────────────────────────────


@mcp.tool()
async def map_metadata_to_worldview_layers(
    dataset: dict[str, str],
    limit: int = 10,
) -> dict[str, Any]:
    """Map one extraction-agent dataset object to NASA Worldview layer IDs.

    Pass a single item from the extraction agent's ``datasets`` array directly.
    The tool reads the exact field names produced by the extraction agent,
    builds a search query, and runs hybrid (vector + BM25) search over the
    in-memory Worldview layer catalog (loaded from worldview_unified_final.json).

    Vector search uses pre-computed embeddings loaded from a companion
    .npy file (created by generate_embeddings.py).  If the file is absent,
    the tool falls back to BM25-only and notes this in the ``search_mode``
    field of each result.

    Args:
        dataset: One dataset object from the extraction agent, e.g.:
            {
              "Satellite data name":    "MODIS Terra",
              "Sensor Name":            "MODIS",
              "Variable / Measurement": "land surface reflectance",
              "Phenomenon":             "burned area wildfire",
              "Science topic":          "fire ecology",
              "Spatial Resolution":     "500m",
              "Temporal Resolution":    "8-day",
              "Acquisition Start Date": "2019",
              "Acquisition End Date":   "2021",
              "Location Coverage":      "Amazon basin",
              "Processing Level":       "L3",
              "How used":               "to map fire-affected areas"
            }
        limit: Max layers to return (default 5).

    Returns:
        matched_layers:           list of layer dicts, each with:
                                    layer_id, display_name, description,
                                    date_range_start, date_range_end,
                                    instrument, platform, tags,
                                    relevance_score, vector_score, bm25_score,
                                    search_mode ("hybrid" | "bm25_only")
        query_used:               natural-language string that was searched
        acquisition_dates_parsed: dates derived from "Acquisition Start Date"
                                  and "Acquisition End Date"
        dataset_echo:             the input dataset (for traceability)
        total_layers_in_catalog:  size of the loaded layer index
    """
    satellite  = (dataset.get("Satellite data name") or "").strip()
    sensor     = (dataset.get("Sensor Name") or "").strip()
    variable   = (dataset.get("Variable / Measurement") or "").strip()
    phenomenon = (dataset.get("Phenomenon") or "").strip()
    topic      = (dataset.get("Science topic") or "").strip()
    location   = (dataset.get("Location Coverage") or "").strip()
    acq_start  = (dataset.get("Acquisition Start Date") or "").strip()
    acq_end    = (dataset.get("Acquisition End Date") or "").strip()

    query = " ".join(p for p in [variable, phenomenon, satellite, sensor, topic, location] if p)
    if not query:
        return {
            "error": (
                "dataset must contain at least one non-empty field among: "
                "'Variable / Measurement', 'Phenomenon', 'Satellite data name', "
                "'Sensor Name', 'Science topic'."
            ),
            "matched_layers": [],
        }

    norm_dates = _normalise_dates([d for d in [acq_start, acq_end] if d])

    try:
        matched = _hybrid_search(query, norm_dates, limit)
    except FileNotFoundError as e:
        return {"error": str(e), "matched_layers": []}

    return {
        "matched_layers": matched,
        "query_used": query,
        "acquisition_dates_parsed": norm_dates,
        "dataset_echo": dataset,
        "total_layers_in_catalog": len(_layers),
    }


# ── Tool 2: validate_temporal_coverage ────────────────────────────────────────


@mcp.tool()
async def validate_temporal_coverage(
    layer_id: str,
    acquisition_start_date: str,
    acquisition_end_date: str,
) -> dict[str, Any]:
    """Validate that a Worldview layer overlaps a requested acquisition window.

    Looks up the layer in the in-memory catalog (loaded from
    worldview_unified_final.json) by exact layer_id and compares the requested
    acquisition window against the layer's date_range_start / date_range_end.

    Args:
        layer_id:          Exact Worldview layer ID, e.g.
                           "MODIS_Terra_SurfaceReflectance_Bands143"
        acquisition_start_date: Start of requested window (YYYY, YYYY-MM, YYYY-MM-DD)
        acquisition_end_date:   End of requested window (YYYY, YYYY-MM, YYYY-MM-DD)

    Returns:
        layer_id:            echoed back
        layer_start:         layer's start date (YYYY-MM-DD) or None
        layer_end:           layer's end date (YYYY-MM-DD) or None
                             None means the layer is currently active
        is_currently_active: True when layer_end is None
        requested_window:    {"start", "end"} after normalization
        overlap_type:        "COMPLETE_OVERLAP" | "PARTIAL_OVERLAP" | "NO_OVERLAP"
        overlap_window:      {"start", "end"} for covered portion, or None
        not_covered_windows: list of {"start", "end", "reason"}
        overall_valid:       True only for COMPLETE_OVERLAP
        source:              "json" | "not_found"
    """
    if not layer_id.strip():
        return {"error": "layer_id is required", "overall_valid": False}
    if not acquisition_start_date.strip() or not acquisition_end_date.strip():
        return {
            "error": "acquisition_start_date and acquisition_end_date are required.",
            "overall_valid": False,
        }

    norm_window = _normalise_dates([acquisition_start_date, acquisition_end_date])
    if len(norm_window) != 2:
        return {
            "error": (
                "Could not parse acquisition_start_date/acquisition_end_date. "
                "Use YYYY, YYYY-MM, or YYYY-MM-DD."
            ),
            "overall_valid": False,
        }

    req_start_str, req_end_str = norm_window
    req_start_dt = _parse_date(req_start_str)
    req_end_dt = _parse_date(req_end_str)
    if req_start_dt is None or req_end_dt is None:
        return {
            "error": "Could not parse normalized acquisition window into valid dates.",
            "overall_valid": False,
        }
    if req_start_dt > req_end_dt:
        return {
            "error": "acquisition_start_date must be <= acquisition_end_date.",
            "overall_valid": False,
            "requested_window": {"start": req_start_str, "end": req_end_str},
        }

    # Ensure the index is loaded
    try:
        _build_index()
    except FileNotFoundError as e:
        return {"error": str(e), "overall_valid": False}

    layer = _layer_id_map.get(layer_id)
    if layer is None:
        return {
            "layer_id": layer_id,
            "error": f"Layer '{layer_id}' not found in the catalog.",
            "overall_valid": False,
            "source": "not_found",
        }

    layer_start_str = layer.get("date_range_start")
    layer_end_str   = layer.get("date_range_end")
    layer_start_dt  = _parse_date(layer_start_str)
    layer_end_dt    = _parse_date(layer_end_str)
    is_currently_active = layer_end_dt is None
    now = datetime.now(UTC)

    # For active layers (no end date), cap coverage at "now" for practical availability.
    effective_layer_end_dt = min(layer_end_dt, now) if layer_end_dt else now
    effective_layer_start_dt = layer_start_dt or datetime.min.replace(tzinfo=UTC)

    overlap_start_dt = max(req_start_dt, effective_layer_start_dt)
    overlap_end_dt = min(req_end_dt, effective_layer_end_dt)
    has_overlap = overlap_start_dt <= overlap_end_dt

    def _fmt(dt: datetime) -> str:
        return dt.date().isoformat()

    overlap_window: dict[str, str] | None = None
    not_covered_windows: list[dict[str, str]] = []
    if has_overlap:
        overlap_window = {"start": _fmt(overlap_start_dt), "end": _fmt(overlap_end_dt)}
        if req_start_dt < overlap_start_dt:
            not_covered_windows.append({
                "start": _fmt(req_start_dt),
                "end": _fmt(overlap_start_dt - timedelta(days=1)),
                "reason": "BEFORE_LAYER_START",
            })
        if req_end_dt > overlap_end_dt:
            reason = "AFTER_LAYER_END"
            if req_end_dt > now and is_currently_active:
                reason = "FUTURE_DATES_NOT_YET_AVAILABLE"
            not_covered_windows.append({
                "start": _fmt(overlap_end_dt + timedelta(days=1)),
                "end": _fmt(req_end_dt),
                "reason": reason,
            })
    else:
        # No overlap: entire requested window is uncovered.
        reason = "OUTSIDE_LAYER_COVERAGE"
        if req_end_dt < effective_layer_start_dt:
            reason = "BEFORE_LAYER_START"
        elif req_start_dt > effective_layer_end_dt:
            reason = "AFTER_LAYER_END"
            if req_start_dt > now and is_currently_active:
                reason = "FUTURE_DATES_NOT_YET_AVAILABLE"
        not_covered_windows.append({
            "start": _fmt(req_start_dt),
            "end": _fmt(req_end_dt),
            "reason": reason,
        })

    if has_overlap and not not_covered_windows:
        overlap_type = "COMPLETE_OVERLAP"
    elif has_overlap:
        overlap_type = "PARTIAL_OVERLAP"
    else:
        overlap_type = "NO_OVERLAP"

    return {
        "layer_id":            layer_id,
        "layer_start":         layer_start_str,
        "layer_end":           layer_end_str,
        "is_currently_active": is_currently_active,
        "requested_window":    {"start": req_start_str, "end": req_end_str},
        "overlap_type":        overlap_type,
        "overlap_window":      overlap_window,
        "not_covered_windows": not_covered_windows,
        "overall_valid":       overlap_type == "COMPLETE_OVERLAP",
        "source":              "json",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
