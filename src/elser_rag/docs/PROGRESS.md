# Progress

## Phases

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Scaffold (config, models, logging, Docker) | ✅ Done | |
| Phase 2 — Ingest pipeline (parser, chunker, enricher) | ✅ Done | |
| Phase 3 — ES indexing (ELSER deploy, ingest pipeline, bulk index) | ✅ Done | |
| Phase 4 — Query pipeline (retriever, generator, RAG orchestrator) | ✅ Done | |
| Phase 5 — API layer (FastAPI endpoints, health, streaming) | ✅ Done | |

## Fixed issues

| # | Status | Description |
|---|--------|-------------|
| 1 | ✅ Fixed | `ImportError: libGL.so.1` — added `libgl1` + `libglib2.0-0` to Dockerfile apt deps. |
| 2 | ✅ Fixed | ES client v9 vs ES server v8 mismatch — upgraded to ES 9.0.1 + removed `<9.0.0` client pin. |
| 3 | ✅ Fixed | Wrong ELSER model ID (hyphens) — corrected to `.elser_model_2` (platform-agnostic for ARM). |
| 4 | ✅ Fixed | `_inference` API requires Enterprise license — added `xpack.license.self_generated.type=trial`. |
| 5 | ✅ Fixed | ES 9.x `service: "elser"` renamed to `service: "elasticsearch"` for native models. |
| 6 | ✅ Fixed | ES 9.x Python client uses `inference_config=` not `body=` for `inference.put`. |
| 7 | ✅ Fixed | ES 9.x ingest pipeline inference processor uses `model_id` not `inference_id`. |
| 8 | ✅ Fixed | ES 9.x removed index-time `boost` on field mappings — removed `"boost": 2.0` from `section_title`. |

## Verified ✅

- `GET /health` returns `{"app": "ok", "elasticsearch": "yellow", "elser_model": "deployed"}`
- App boots cleanly, ELSER inference endpoint deployed, indices created

## Next up
- [X] End-to-end smoke test: ingest 1 PDF, run 1 query via `/query`
- [X] Tune enrichment prompt if context prefix quality is low
- [ ] Add `pytest` integration tests for ingest + query flows
