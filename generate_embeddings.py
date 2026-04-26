"""
Generate and save vector embeddings for worldview_unified_final.json.

Embeddings are stored in a single companion file (NOT inside the JSON):
  <stem>_embeddings.npz  — numpy archive with two named arrays:
                            "embeddings": float32 (N, D)
                            "ids":        str     (N,)   ← layer_id per row

Bundling IDs and vectors in one .npz means ordering is self-contained —
no separate index file to lose or go out of sync.

Keeping embeddings out of the JSON makes the source file diffable and
human-readable, and avoids a ~37 MB float-array payload in a text file.

Checkpoints after every batch so it can be safely re-run if interrupted —
layers whose IDs already appear in the .npz are skipped.

Usage:
    uv run python generate_embeddings.py            # embed everything
    uv run python generate_embeddings.py --dry-run  # estimate only, no API calls

    # or point at a different file:
    WORLDVIEW_JSON_PATH=my_layers.json uv run python generate_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL       = "text-embedding-3-small"
COST_PER_M_TOKENS = 0.02          # USD, text-embedding-3-small
BATCH_SIZE        = 100           # layers per API call (max 2 048; 100 keeps latency low)
RETRY_DELAY       = 5             # seconds to wait after a rate-limit error
MAX_RETRIES       = 3

_DEFAULT_JSON = Path(__file__).parent / "worldview_unified_final.json"
JSON_PATH = Path(os.getenv("WORLDVIEW_JSON_PATH", str(_DEFAULT_JSON)))

# Companion file lives next to the JSON, named after it
_stem           = Path(str(JSON_PATH.with_suffix("")))
EMBEDDINGS_PATH = Path(str(_stem) + "_embeddings.npz")


# ── Text builder (must match server.py _layer_text) ───────────────────────────

def _layer_text(layer: dict) -> str:
    fields = [
        ("layer_id",    layer.get("layer_id", "")),
        ("name",        layer.get("display_name", "")),
        ("description", layer.get("description", "")),
        ("tags",        " ".join(layer.get("tags", []))),
        ("instrument",  layer.get("instrument", "")),
        ("platform",    layer.get("platform", "")),
        ("measurement", layer.get("measurement_id", "")),
    ]
    return " ".join(f"{k}: {v}" for k, v in fields if v)


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens(texts: list[str]) -> list[int]:
    """Return per-text token counts. Uses tiktoken if available, else estimates."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return [len(enc.encode(t)) for t in texts]
    except ImportError:
        # ~4 chars per token is a reasonable approximation
        return [max(1, len(t) // 4) for t in texts]


# ── Dry-run report ────────────────────────────────────────────────────────────

def _dry_run(layers: list[dict], done: set[str]) -> None:
    todo = [l for l in layers if l.get("layer_id") not in done]
    texts = [_layer_text(l) for l in todo]
    token_counts = _count_tokens(texts)

    total_tokens = sum(token_counts)
    estimated_cost = total_tokens / 1_000_000 * COST_PER_M_TOKENS
    n_batches = math.ceil(len(todo) / BATCH_SIZE)

    try:
        import tiktoken  # noqa: F401
        token_method = "tiktoken (cl100k_base)"
    except ImportError:
        token_method = "estimate (~4 chars/token) — install tiktoken for exact counts"

    print()
    print("── Dry-run estimate ─────────────────────────────────────────")
    print(f"  JSON path:          {JSON_PATH}")
    print(f"  Layers in catalog:  {len(layers)}")
    print(f"  Already embedded:   {len(done)}")
    print(f"  To embed:           {len(todo)}")
    print()
    if token_counts:
        print(f"  Tokens per layer:")
        print(f"    min   {min(token_counts):>8,}")
        print(f"    max   {max(token_counts):>8,}")
        print(f"    avg   {total_tokens // len(token_counts):>8,}")
        print(f"    total {total_tokens:>8,}")
    print()
    print(f"  Token counting:     {token_method}")
    print(f"  Model:              {EMBED_MODEL}  @ ${COST_PER_M_TOKENS}/1M tokens")
    print(f"  Estimated cost:     ${estimated_cost:.4f}")
    print(f"  API batches:        {n_batches}  (batch size {BATCH_SIZE})")
    print(f"  Output:             {EMBEDDINGS_PATH}")
    print("─────────────────────────────────────────────────────────────")
    print()


# ── Batched embedding with retry ──────────────────────────────────────────────

def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            # API guarantees order matches input
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(f"Embedding attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY}s …")
            time.sleep(RETRY_DELAY)
    raise RuntimeError("unreachable")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"JSON not found: {JSON_PATH}")

    logger.info(f"Loading {JSON_PATH} …")
    with open(JSON_PATH) as f:
        data = json.load(f)

    layers: list[dict] = data.get("layers", [])
    logger.info(f"  {len(layers)} layers total")

    # Load existing embeddings to support checkpointing
    done: dict[str, list[float]] = {}
    if EMBEDDINGS_PATH.exists():
        archive = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        for lid, vec in zip(archive["ids"], archive["embeddings"]):
            done[str(lid)] = vec.tolist()
        logger.info(f"  {len(done)} already embedded (loaded from {EMBEDDINGS_PATH.name})")

    if dry_run:
        _dry_run(layers, set(done))
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env or export it.")

    todo = [l for l in layers if l.get("layer_id") not in done]
    logger.info(f"  {len(todo)} to process")

    if not todo:
        logger.info("Nothing to do — all layers already have embeddings.")
        return

    client = OpenAI(api_key=api_key)

    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Embedding batches"):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        texts = [_layer_text(l) for l in batch]
        embeddings = _embed_batch(client, texts)

        for layer, embedding in zip(batch, embeddings):
            done[layer["layer_id"]] = embedding

        # Checkpoint: preserve original JSON layer order, save after every batch
        ordered_ids = [l["layer_id"] for l in layers if l.get("layer_id") in done]
        matrix = np.array([done[lid] for lid in ordered_ids], dtype=np.float32)
        np.savez(EMBEDDINGS_PATH, embeddings=matrix, ids=np.array(ordered_ids))

    logger.info(f"Done. {len(todo)} layers embedded.")
    logger.info(f"  Embeddings: {EMBEDDINGS_PATH}  shape={matrix.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for Worldview layers.")
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Estimate token count and cost without calling the API.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
