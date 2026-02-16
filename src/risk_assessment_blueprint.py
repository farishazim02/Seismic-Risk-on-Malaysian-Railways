"""Blueprint pipeline for Malaysia rail seismic after-shock risk assessment.

This script is intentionally modular so it can be moved into a notebook or
kept as a .py production pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# External geospatial/ML packages are expected in project environment.
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import LineString
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Config:
    rail_path: str = "data/rail_segments.gpkg"
    rail_layer: str = "rail_segments"
    eq_path: str = "data/usgs_earthquakes_2021_2025_m3p5.geojson"

    # User-provided scales
    segment_length_m: int = 250
    vs30_buffer_m: int = 1000
    pga_resolution_m: int = 10118
    slope_resolution_m: int = 100

    # Use local UTM zone that best fits your alignment
    target_crs: str = "EPSG:32647"


def load_core_data(cfg: Config) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    rails = gpd.read_file(cfg.rail_path, layer=cfg.rail_layer).to_crs(cfg.target_crs)
    eq = gpd.read_file(cfg.eq_path).to_crs(cfg.target_crs)
    return rails, eq


def _split_line_every_n_m(line: LineString, n_m: float) -> list[LineString]:
    """Split a LineString into ~equal chunks of length n_m."""
    if line.length <= n_m:
        return [line]
    distances = np.arange(0, line.length, n_m)
    parts: list[LineString] = []
    for start in distances:
        end = min(start + n_m, line.length)
        seg = LineString([line.interpolate(start), line.interpolate(end)])
        parts.append(seg)
    return parts


def segment_rails(rails: gpd.GeoDataFrame, segment_length_m: int) -> gpd.GeoDataFrame:
    records = []
    for idx, row in rails.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == "LineString":
            pieces = _split_line_every_n_m(geom, segment_length_m)
        elif geom.geom_type == "MultiLineString":
            pieces = []
            for part in geom.geoms:
                pieces.extend(_split_line_every_n_m(part, segment_length_m))
        else:
            continue

        for i, piece in enumerate(pieces):
            rec = row.drop(labels=["geometry"]).to_dict()
            rec["parent_idx"] = idx
            rec["part_idx"] = i
            rec["geometry"] = piece
            records.append(rec)

    out = gpd.GeoDataFrame(records, crs=rails.crs)
    out["segment_id"] = [f"SEG_{i:08d}" for i in range(len(out))]
    return out


def add_distance_to_latest_events(
    seg: gpd.GeoDataFrame,
    eq: gpd.GeoDataFrame,
    top_n_recent: int = 30,
) -> gpd.GeoDataFrame:
    eq = eq.copy()
    eq["time"] = pd.to_datetime(eq["time"], errors="coerce")
    eq_recent = eq.sort_values("time", ascending=False).head(top_n_recent)

    seg_cent = seg.copy()
    seg_cent["geometry"] = seg_cent.geometry.centroid

    # nearest event distance
    nearest = gpd.sjoin_nearest(
        seg_cent,
        eq_recent[["time", "mag", "depth", "geometry"]],
        how="left",
        distance_col="dist_eq_m",
    )

    seg["dist_eq_km"] = nearest["dist_eq_m"] / 1000.0
    seg["eq_mag"] = nearest["mag"]
    seg["eq_depth_km"] = nearest["depth"]
    return seg


def add_raster_stats(
    seg: gpd.GeoDataFrame,
    raster_path: str,
    feature_prefix: str,
    buffer_m: int | None = None,
) -> gpd.GeoDataFrame:
    geom = seg.geometry if buffer_m is None else seg.geometry.buffer(buffer_m)
    with rasterio.open(raster_path) as src:
        stats = zonal_stats(
            geom,
            src.read(1),
            affine=src.transform,
            nodata=src.nodata,
            stats=["mean", "min", "max", "std", "percentile_90"],
        )

    out = seg.copy()
    for k in ["mean", "min", "max", "std", "percentile_90"]:
        out[f"{feature_prefix}_{k}"] = [s.get(k) for s in stats]
    return out


def add_quality_metadata(seg: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    out = seg.copy()
    out["vs30_source_resolution_m"] = 1000
    out["pga_source_resolution_m"] = cfg.pga_resolution_m
    out["slope_source_resolution_m"] = cfg.slope_resolution_m

    # Lower value = lower confidence in fine-scale precision.
    out["pga_confidence_weight"] = 1.0 / (1.0 + cfg.pga_resolution_m / cfg.segment_length_m)
    out["slope_confidence_weight"] = 1.0 / (1.0 + cfg.slope_resolution_m / cfg.segment_length_m)
    return out


def rule_based_risk_score(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = df.copy()
    # Example heuristic score (replace with engineering-calibrated weights)
    out["risk_score"] = (
        0.25 * (1 / (out["dist_eq_km"].clip(lower=1)))
        + 0.20 * out["eq_mag"].fillna(0)
        + 0.15 * out.get("pga_mean", pd.Series(np.nan, index=out.index)).fillna(0)
        + 0.15 * out.get("slope_percentile_90", pd.Series(np.nan, index=out.index)).fillna(0)
        + 0.15 * (760 - out.get("vs30_mean", pd.Series(np.nan, index=out.index)).fillna(760)).clip(lower=0)
        + 0.10 * out.get("rain_7d_mm", pd.Series(0, index=out.index)).fillna(0)
    )

    q1, q2 = out["risk_score"].quantile([0.6, 0.85])
    out["risk_class"] = np.select(
        [out["risk_score"] >= q2, out["risk_score"] >= q1],
        ["High", "Medium"],
        default="Low",
    )
    return out


def fit_baseline_ml(df: pd.DataFrame, label_col: str = "risk_class") -> tuple[Pipeline, str]:
    features = [
        "dist_eq_km",
        "eq_mag",
        "eq_depth_km",
        "vs30_mean",
        "pga_mean",
        "slope_mean",
        "slope_percentile_90",
        "rain_3d_mm",
        "rain_7d_mm",
        "asset_type",
    ]
    available = [c for c in features if c in df.columns]

    X = df[available]
    y = df[label_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
    )

    # Keep both models available; RF used by default.
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    # model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)
    return clf, report


def build_priority_list(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    out = df.sort_values("risk_score", ascending=False).head(top_n).copy()
    out["recommended_action"] = np.select(
        [out["risk_class"] == "High", out["risk_class"] == "Medium"],
        ["Inspect within 24h", "Inspect within 48h"],
        default="Routine inspection",
    )
    return out


def main() -> None:
    cfg = Config()
    rails, eq = load_core_data(cfg)

    seg = segment_rails(rails, cfg.segment_length_m)
    seg = add_distance_to_latest_events(seg, eq)

    # Example calls (uncomment when rasters are ready)
    # seg = add_raster_stats(seg, "data/vs30.tif", "vs30", buffer_m=cfg.vs30_buffer_m)
    # seg = add_raster_stats(seg, "data/pga.tif", "pga")
    # seg = add_raster_stats(seg, "data/slope.tif", "slope")

    seg = add_quality_metadata(seg, cfg)
    scored = rule_based_risk_score(seg)

    # Save operational outputs
    scored.to_file("data/segment_risk_latest.gpkg", layer="segment_risk", driver="GPKG")
    build_priority_list(scored).to_csv("data/priority_inspection_list.csv", index=False)

    print("Pipeline completed. Outputs written to data/.")


if __name__ == "__main__":
    main()
