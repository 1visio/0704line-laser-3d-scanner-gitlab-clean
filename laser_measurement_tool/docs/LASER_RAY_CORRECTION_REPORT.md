# Frozen Daheng C1 ray-depth correction

## Conclusions

- `C0_SEMANTIC_MATCH: PASS` — online `quadratic_graph.yaml` and the Frozen C0 were compared after YAML parsing. `model_type`, dependent/independent axes and order, equation, normalization center/scale, all six coefficients, and `z_valid_range_mm` are identical. No refit or C0 value change was performed.
- `C1_OFF_REGRESSION: PASS` — the disabled path uses the original C0 lambda and produces array-identical camera/ground points and filtering results.
- `C1_FROZEN_REPRODUCTION: PASS` — the copied `frozen_c1_4k.json` is byte-content equivalent after line-ending normalization to the supplied Frozen C1. Runtime uses the stored Full-36 PCA center/axis and exact cubic B-spline parameters; it does not recompute PCA or fit parameters.
- `C1_CLAMP: PASS` — `s_raw` is clipped to the frozen `[domain_min, domain_max]` before `BSpline(..., extrapolate=False)` evaluation; both endpoint cases are covered by tests.
- `NON_DAHENG_REGRESSION: PASS` — default and legacy system configurations have no C1 path and retain `enable_laser_ray_correction=False`.

## Data flow

`calibration.laser_ray_correction` is an optional path parsed by `app_config.py`. Both explicit `load_calibration_files(..., laser_ray_correction=...)` and the optional `files.laser_ray_correction` manifest entry call the same Frozen C1 loader. The core applies C1 only after `_intersect_laser_surface()` returns `lambda_c0` and immediately before `points_camera = rays * lambda_final`.

The manifest remains `schema_version: 1`; the new manifest file entry is optional, so old manifests remain valid. C1 is not connected to `ground_u_compensation`.

The audited call paths are: `scan_offline.py -> FramePipeline.run_frame()`, online `FramePipeline.run_frame()`, GUI `_run_measurement()`/cached reconstruction, and standalone tools that call `reconstruct_uv_to_ground()` after explicit calibration loading. All converge on the same core function; no caller contains a C1 formula.

## Tests

- `pytest -q tests/test_laser_ray_correction.py`: 9 passed.
- `pytest -q tests/test_reconstructor.py tests/test_config_loader.py tests/test_config_loader_files.py tests/test_app_config.py tests/test_calibration_manifest.py`: 54 passed, 6 subtests passed.
- `pytest -q tests/test_online_core.py`: 13 passed.
- Combined C1/reconstruction/configuration/online selection: 76 passed, 6 subtests passed.
- Full suite excluding one timing-sensitive existing online throttle test: 210 passed, 1 deselected, 10 subtests passed.
- The excluded throttle test passed in isolation; it had one intermittent timing failure during the unfiltered full-suite run and was not changed.
