"""
Quick local tests for the two benchmark MCP tools.
Runs the tool functions directly (no MCP transport overhead).

Usage:
    uv run python test_local.py
"""

import asyncio
import json

from server import search_worldview_layers, validate_temporal_coverage


def _print(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


async def main() -> None:
    # ── Test 1: metadata-derived search — fire / burned area paper ────────────
    print("\n[1] search_worldview_layers — metadata-derived query (fire paper)")
    result1 = await search_worldview_layers(
        query="land surface reflectance burned area wildfire MODIS Terra fire ecology 500m 8-day Amazon basin L3",
        limit=3,
    )
    _print("Fire paper → matched layers", result1)

    # ── Test 2: metadata-derived search — SST / coral bleaching paper ─────────
    print("\n[2] search_worldview_layers — metadata-derived query (SST paper)")
    result2 = await search_worldview_layers(
        query="sea surface temperature coral bleaching MODIS Aqua oceanography 4km daily Great Barrier Reef L3",
        limit=3,
    )
    _print("SST paper → matched layers", result2)

    # ── Test 3: free-text user-style query (benchmark query validation) ───────
    print("\n[3] search_worldview_layers — free-text user-style query")
    result3 = await search_worldview_layers(
        query="Show me daily sea surface temperature anomalies in the Gulf of Mexico for summer 2020",
        limit=3,
    )
    _print("User-style query → matched layers", result3)

    # ── Test 4: broad search ──────────────────────────────────────────────────
    print("\n[4] search_worldview_layers — broad topic search")
    result4 = await search_worldview_layers(
        query="vegetation index drought monitoring sub-Saharan Africa",
        limit=3,
    )
    _print("Broad search → matched layers", result4)

    # ── Test 5: temporal validation — active layer, complete overlap ───────────
    layer_id = "VIIRS_NOAA20_CorrectedReflectance_TrueColor"
    print(f"\n[5] validate_temporal_coverage — active layer complete overlap ({layer_id})")
    result5 = await validate_temporal_coverage(
        layer_id=layer_id,
        acquisition_start_date="2020-06-15",
        acquisition_end_date="2022-01-01",
    )
    _print(f"{layer_id} coverage check", result5)

    # ── Test 6: temporal validation — likely ended mission, partial/no overlap ─
    layer_id_old = "MODIS_Terra_CorrectedReflectance_TrueColor"
    print(f"\n[6] validate_temporal_coverage — old layer overlap assessment ({layer_id_old})")
    result6 = await validate_temporal_coverage(
        layer_id=layer_id_old,
        acquisition_start_date="2003-01-01",
        acquisition_end_date="2025-06-01",
    )
    _print(f"{layer_id_old} coverage check", result6)

    # ── Test 7: future-only window (expected no overlap for active layers) ─────
    print(f"\n[7] validate_temporal_coverage — future-only window")
    result7 = await validate_temporal_coverage(
        layer_id=layer_id,
        acquisition_start_date="2099-01-01",
        acquisition_end_date="2099-12-01",
    )
    _print("Future date check", result7)

    # ── Test 8: unknown layer (should error gracefully) ────────────────────────
    print(f"\n[8] validate_temporal_coverage — unknown layer_id")
    result8 = await validate_temporal_coverage(
        layer_id="NONEXISTENT_LAYER_XYZ",
        acquisition_start_date="2020-01-01",
        acquisition_end_date="2020-01-31",
    )
    _print("Unknown layer check", result8)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done.  Review relevance_score / overlap_type in each result.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
