# Seismic After-shock Risk Assessment Workplan (Malaysia Rail)

## 1) Recommended approach from your current stage
You already have rail geometries and earthquake events. The next step is to convert this into a **repeatable asset-level risk pipeline**:

1. Build a master rail segment table (250 m segments, unique `segment_id`).
2. Join multi-source hazard factors (VS30, PGA, slope, rainfall, distance-to-event/fault).
3. Add vulnerability factors (asset class, age, embankment/cut, bridge/tunnel flags).
4. Build two outputs:
   - **Interpretable Risk Score** (for immediate operations)
   - **ML-based Risk Classifier** (for performance and future automation)
5. Publish GIS layers + dashboard-oriented CSVs for inspection prioritization.

---

## 2) Spatial harmonisation decisions (for your stated resolutions)
Given:
- VS30 buffer = 1000 m
- Rail segmentation = 250 m
- PGA resolution = 10,118 m
- Slope resolution = 100 m

Use these rules:

1. **Reference geometry**: keep 250 m rail segments as the base analysis unit.
2. **CRS**: project all data to a metric CRS (e.g., `EPSG:32647` or `EPSG:32648` by corridor location; avoid degree-based distance math).
3. **VS30**: aggregate within 1000 m around each segment midpoint or buffered segment; store `vs30_mean`, `vs30_min`, `vs30_std`.
4. **PGA (coarse)**: never treat as precise local value. Use zonal mean + explicit uncertainty flag (`pga_resolution_m=10118`).
5. **Slope (fine)**: sample both along the segment and near-side buffer (e.g., 30–100 m). Store `slope_p90` to capture local instability pockets.
6. **Multi-resolution quality fields** (important): add per-feature metadata columns:
   - `source_resolution_m`
   - `distance_to_source_m`
   - `data_age_days`
   - `confidence_weight`

This lets decision-makers understand confidence, not only risk magnitude.

---

## 3) Suggested risk factors (expanded)
Besides your listed factors, add these high-value predictors:

### Hazard intensity
- Distance to latest after-shock epicentre (`dist_eq_km`)
- Event magnitude and depth (`eq_mag`, `eq_depth_km`)
- Distance to active faults (`dist_fault_km`)
- PGA (`pga_mean`)
- Site amplification proxy from VS30 (`vs30_mean`, category)

### Terrain and hydrology
- Slope statistics (`slope_mean`, `slope_p90`)
- Relative relief / elevation range in 250–500 m window
- Drainage density or proximity to river/floodplain
- 3-day / 7-day cumulative rainfall prior to event (`rain_3d_mm`, `rain_7d_mm`)

### Asset vulnerability
- Asset type (`embankment`, `bridge`, `tunnel`, `at-grade`)
- Asset age / last renewal year
- Known poor subgrade sections
- Curvature and gradient (track geometry sensitivity)
- Historical defects and maintenance frequency

### Exposure / consequence
- Passenger/freight criticality index
- Nearby station/yard operational importance
- Availability of detour route

---

## 4) Target outputs for operations
Create these deliverables every analysis cycle:

1. `segment_risk_latest.gpkg` (GIS layer)
2. `priority_inspection_list.csv` (top N high-risk segments)
3. `risk_summary_by_line.csv` (management-level view)
4. Dashboard fields:
   - risk class (`Low`, `Medium`, `High`)
   - key drivers (`top_3_drivers`)
   - confidence indicator (`High/Med/Low`)
   - recommended action (`Inspect within 24h`, `48h`, `Routine`)

---

## 5) Modelling strategy
Use a two-track strategy:

1. **Baseline transparent model**
   - Weighted scoring model with engineering-informed weights
   - Useful immediately even with limited labeled failures

2. **Supervised ML model**
   - Start with Logistic Regression + Random Forest
   - Use class imbalance handling (`class_weight='balanced'` or SMOTE)
   - Evaluate with PR-AUC, recall@top-k, confusion matrix by risk class

If labeled damage outcomes are sparse, train on proxy labels first (historical defect occurrences post-events), then refine when real inspection data accumulates.

---

## 6) Validation and governance checklist
- Temporal split (train on older period, validate on newer events)
- Spatial split (hold out one corridor/line)
- Drift monitoring (input distributions and risk-class proportions)
- Explainability: SHAP or feature importances in monthly report
- Human-in-loop review for top 1–5% highest risk segments

---

## 7) 6-week execution plan
1. **Week 1**: Data audit, CRS harmonisation, segment ID standardization
2. **Week 2**: Feature engineering (hazard + terrain + vulnerability)
3. **Week 3**: Baseline weighted risk score and threshold calibration
4. **Week 4**: Logistic/RandomForest experiments and evaluation
5. **Week 5**: GIS outputs + dashboard tables + inspection ranking
6. **Week 6**: Validation report, SOP draft, handover package

---

## 8) Immediate next action
Use `src/risk_assessment_blueprint.py` as the starter pipeline, then adapt file paths and available datasets in your environment.
