"""
Quick local tests for the two benchmark MCP tools.
Runs the tool functions directly (no MCP transport overhead).

Usage:
    uv run python test_local.py
"""

import asyncio
import json

from server import map_metadata_to_worldview_layers, validate_temporal_coverage


def _print(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


async def main() -> None:
    # ── Test 1: typical fire paper (exact extraction-agent output shape) ──────
    print("\n[1] map_metadata_to_worldview_layers — fire / burned area paper")
    result1 = await map_metadata_to_worldview_layers(
        dataset={
            "Satellite data name":    "MODIS Terra",
            "Sensor Name":            "MODIS",
            "Variable / Measurement": "land surface reflectance",
            "Phenomenon":             "burned area wildfire",
            "Science topic":          "fire ecology",
            "Spatial Resolution":     "500m",
            "Temporal Resolution":    "8-day",
            "Acquisition Start Date": "2020-08",
            "Acquisition End Date":   "2020-09",
            "Location Coverage":      "Amazon basin",
            "Processing Level":       "L3",
            "How used":               "to map fire-affected areas",
        },
        limit=3,
    )
    _print("Fire paper → matched layers", result1)

    # ── Test 2: ocean / SST paper ─────────────────────────────────────────────
    print("\n[2] map_metadata_to_worldview_layers — SST / coral bleaching paper")
    result2 = await map_metadata_to_worldview_layers(
        dataset={
            "Satellite data name":    "Aqua",
            "Sensor Name":            "MODIS",
            "Variable / Measurement": "sea surface temperature",
            "Phenomenon":             "coral bleaching",
            "Science topic":          "oceanography",
            "Spatial Resolution":     "4km",
            "Temporal Resolution":    "daily",
            "Acquisition Start Date": "2016",
            "Acquisition End Date":   "2017",
            "Location Coverage":      "Great Barrier Reef",
            "Processing Level":       "L3",
            "How used":               "to detect thermal stress events",
        },
        limit=3,
    )
    _print("SST paper → matched layers", result2)

    # ── Test 3: temporal validation — active layer, complete overlap ───────────
    # Use a layer_id from your Weaviate collection; VIIRS is a safe bet
    layer_id = "VIIRS_NOAA20_CorrectedReflectance_TrueColor"
    print(f"\n[3] validate_temporal_coverage — active layer complete overlap ({layer_id})")
    result3 = await validate_temporal_coverage(
        layer_id=layer_id,
        acquisition_start_date="2020-06-15",
        acquisition_end_date="2022-01-01",
    )
    _print(f"{layer_id} coverage check", result3)

    # ── Test 4: temporal validation — likely ended mission, partial/no overlap ─
    layer_id_old = "MODIS_Terra_CorrectedReflectance_TrueColor"
    print(f"\n[4] validate_temporal_coverage — old layer overlap assessment ({layer_id_old})")
    result4 = await validate_temporal_coverage(
        layer_id=layer_id_old,
        acquisition_start_date="2003-01-01",
        acquisition_end_date="2025-06-01",
    )
    _print(f"{layer_id_old} coverage check", result4)

    # ── Test 5: future-only window (expected no overlap for active layers) ─────
    print(f"\n[5] validate_temporal_coverage — future-only window")
    result5 = await validate_temporal_coverage(
        layer_id=layer_id,
        acquisition_start_date="2099-01-01",
        acquisition_end_date="2099-12-01",
    )
    _print("Future date check", result5)

    # ── Test 6: unknown layer (should error gracefully) ────────────────────────
    print(f"\n[6] validate_temporal_coverage — unknown layer_id")
    result6 = await validate_temporal_coverage(
        layer_id="NONEXISTENT_LAYER_XYZ",
        acquisition_start_date="2020-01-01",
        acquisition_end_date="2020-01-31",
    )
    _print("Unknown layer check", result6)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done.  Check 'source' field in each result:")
    print("    weaviate     → connected successfully")
    print("    json_fallback → Weaviate unreachable, used local JSON")
    print("    none         → layer not found anywhere")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
