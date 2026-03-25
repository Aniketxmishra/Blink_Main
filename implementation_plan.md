# Enterprise-Ready LLM Support for Blink

## Current State

| Metric | Value | Notes |
|--------|-------|-------|
| LLM MAPE (overall) | **11.5%** | Only 32 data points, 2 models (gpt2, gpt2-medium) |
| gpt2-medium MAPE | 10.2% | 5 test samples |
| gpt2 MAPE | 13.7% | 3 test samples |
| CNN MAPE | ~14% | 296 data points, 10+ architectures |
| LLM models profiled | 2 | gpt2 (124M), gpt2-medium (355M) |
| Quantization configs | fp32 only | fp16 too slow on CPU, int8/int4 needs CUDA |
| GPU hardware profiled | 0 for LLM | All LLM data is CPU-only |

> [!CAUTION]
> The 11.5% MAPE is **misleadingly optimistic**: it's trained and tested on the same 2 model families running on CPU. Real enterprise usage involves diverse LLMs (1B–70B params) on GPU hardware with inference engines (vLLM, TGI). The model has never seen any of this.

---

## Proposed Changes

### Phase 1: Fix Existing Broken Plumbing (Priority: Immediate)

These are bugs in the current code that prevent LLM predictions from working end-to-end.

---

#### [MODIFY] [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py)

**Problem:** `feature_cols` (line 69-86) is missing the 4 LLM columns (`vocab_size`, `seq_len`, `kv_cache_size_mb`, `quantization_bits`). The XGBoost model was retrained with these features, but the predictor still constructs feature vectors without them → **silent mismatch → garbage predictions for LLMs.**

**Also:** [predict_for_custom_model()](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py#198-265) (line 198) doesn't accept `seq_len` or `quantization_bits` parameters, and hardcodes `input_shape=(3,224,224)` even for LLMs. It doesn't call `ModelAnalyzer.extract_features()` — it calls the old `ModelFeatureExtractor`.

**Fix:**
1. Add `vocab_size`, `seq_len`, `kv_cache_size_mb`, `quantization_bits` to `self.feature_cols`
2. Update [predict_for_custom_model()](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py#198-265) to accept `seq_len` and `quantization_bits` kwargs
3. Detect HuggingFace models and route to `ModelAnalyzer.extract_features()` instead of `ModelFeatureExtractor`
4. Return `prefill_memory_mb` and `decode_memory_mb` from the KV cache math in the result dict

---

#### [MODIFY] [dashboard.py](file:///c:/Aniket/review%20blink/Neusight/dashboard.py)

**Problem:** The dashboard only offers "Custom CNN", "Pre-trained Models" (CNNs), and "PyTorch Code (GNN)". There is no way to select an LLM, specify sequence length, or choose quantization.

**Fix:**
1. Add "HuggingFace LLM" as a 4th model type in [show_prediction_page()](file:///c:/Aniket/review%20blink/Neusight/dashboard.py#115-520) 
2. Add UI controls for: HF model ID text input, sequence length slider (64–4096), quantization radio (fp32/fp16/int8/int4)
3. Show LLM-specific metrics: Prefill time (TTFT), Decode time (TPOT), KV Cache size, Prefill memory, Decode memory
4. Add a "Token Generation Timeline" Plotly chart showing TTFT + N×TPOT for configurable output lengths

---

#### [MODIFY] [prediction_api.py](file:///c:/Aniket/review%20blink/Neusight/prediction_api.py)

**Problem:** The REST API doesn't accept LLM parameters.

**Fix:**
1. Add `seq_len`, `quantization_bits`, and `model_id` fields to the prediction request schema
2. Route HuggingFace model IDs through the LLM feature extraction path
3. Return `prefill_time_ms`, `decode_time_ms`, `kv_cache_size_mb` in the response

---

### Phase 2: Scale the Training Data (Priority: High)

The model has only seen 2 architectures on CPU. Enterprise customers will ask about LLaMA-3, Mistral, Qwen on A100/H100.

---

#### [MODIFY] [scripts/collect_data.py](file:///c:/Aniket/review%20blink/Neusight/scripts/collect_data.py)

**Changes:**
1. Add more models to `CAUSAL_LLM_CONFIGS`: `microsoft/phi-2` (2.7B), `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `meta-llama/Llama-3.2-1B` (if accessible)
2. Fix the `device_map` issue so int8/int4 profiling works when CUDA is available
3. Add a `--gpu-name` override flag so data collected on cloud GPUs gets tagged with the correct hardware specs

#### [NEW] [scripts/colab_llm_collector.py](file:///c:/Aniket/review%20blink/Neusight/scripts/colab_llm_collector.py)

A Google Colab-optimized script (runs on free T4 GPU) that:
- Profiles the LLM configs across fp32, fp16, int8, int4
- Collects real GPU timing with `torch.cuda.Event` (not `time.perf_counter`)
- Saves results as CSV that can be downloaded and merged into `data/enriched/`

#### [NEW] [data/enriched/synthetic_llm_scaling.csv](file:///c:/Aniket/review%20blink/Neusight/data/enriched/synthetic_llm_scaling.csv)

Generate synthetic scaling data using the roofline model:
- For each GPU in `GPU_SPECS` (H100, A100, T4, L4, RTX 3090, RTX 4090)
- For each model size (1B, 3B, 7B, 13B, 70B params)
- For each precision (fp32, fp16, int8, int4)
- Calculate theoretical prefill time = FLOPs / effective_TFLOPS
- Calculate theoretical decode time = model_bytes / memory_bandwidth
- This gives ~720 synthetic rows that teach the model about hardware scaling

---

### Phase 3: Separate Prefill & Decode Prediction Models (Priority: High)

Currently XGBoost predicts a single `execution_time_ms`. For LLMs, prefill and decode have fundamentally different bottlenecks (compute-bound vs memory-bandwidth-bound).

---

#### [MODIFY] [prediction_model.py](file:///c:/Aniket/review%20blink/Neusight/prediction_model.py)

**Changes:**
1. In `train_models()`, train **3 separate XGBoost models**:
   - `exec_time_model` — existing CNN latency predictor (unchanged)
   - `prefill_time_model` — trained on LLM rows where target = `prefill_time_ms`
   - `decode_time_model` — trained on LLM rows where target = `decode_time_ms`
2. Save all three models + their quantile bounds to `models/`
3. In `prepare_features()`, include `is_llm` as a feature so the unified model can also learn the distinction

#### [MODIFY] [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py)

Load and use the prefill/decode models when `is_llm=True`:
```python
if features.get('is_llm'):
    result['prefill_time_ms'] = self.prefill_model.predict(...)
    result['decode_time_ms']  = self.decode_model.predict(...)
    result['ttft_ms'] = result['prefill_time_ms']
    result['tpot_ms'] = result['decode_time_ms']
else:
    result['exec_time_ms'] = self.model.predict(...)
```

---

### Phase 4: Hardware Recommender (Priority: Medium)

The enterprise killer feature: "Given my model and SLA, which GPU should I rent?"

---

#### [NEW] [hardware_recommender.py](file:///c:/Aniket/review%20blink/Neusight/hardware_recommender.py)

Uses `GPU_SPECS` from `gpu_info.py` + the trained prefill/decode models to:
1. Take inputs: model_name, batch_size, seq_len, quantization_bits, SLA constraints (max TTFT, max TPOT, max memory)
2. Simulate the model on every GPU in the spec table
3. Return a ranked list: cheapest GPU that meets ALL SLA constraints
4. Include estimated cloud cost/hour (AWS/GCP pricing table)

#### [MODIFY] [dashboard.py](file:///c:/Aniket/review%20blink/Neusight/dashboard.py)

Add a "Hardware Recommender" page to the sidebar navigation.

---

### Phase 5: Tests & Validation (Priority: High)

---

#### [NEW] [tests/test_llm_support.py](file:///c:/Aniket/review%20blink/Neusight/tests/test_llm_support.py)

End-to-end test suite covering:
1. `ModelAnalyzer.extract_features()` with gpt2 → verify `is_llm=True`, `kv_cache_size_mb > 0`, `vocab_size == 50257`
2. `GPUPredictor.predict()` with LLM features → verify returns `prefill_time_ms`, `decode_time_ms`
3. `GPUPredictor.predict_for_custom_model(gpt2, batch_size=1, seq_len=128)` → verify result dict has LLM fields
4. `get_hardware_specs("A100", quantization_bits=8)` → verify `effective_tflops == 19.5 * 4`
5. KV cache math sanity: for gpt2 (12 layers, 768 hidden, 12 heads) at bs=1 seq=512 fp32, verify `kv_cache_size_mb ≈ 36 MB`

---

## Verification Plan

### Automated Tests

```bash
# Run the new LLM test suite
python -m pytest tests/test_llm_support.py -v

# Verify prediction_model.py trains without errors and produces 3 model files
python prediction_model.py
# Check: models/prefill_time_model.joblib exists
# Check: models/decode_time_model.joblib exists

# Verify the CI/CD script works end-to-end
python blink_github_action.py --model-name gpt2 --batch-size 1 --seq-len 128 --sla-latency-ms 5000 --sla-memory-mb 8000
# Expected: exit code 0 (PASS)

python blink_github_action.py --model-name gpt2 --batch-size 1 --seq-len 128 --sla-latency-ms 1
# Expected: exit code 1 (FAIL — 1ms is impossible)
```

### Manual Verification (Dashboard)

1. Run `streamlit run dashboard.py`
2. Select "HuggingFace LLM" from the model type dropdown
3. Enter `gpt2` as the model ID
4. Set sequence length to 256, quantization to fp16
5. Click "Predict GPU Usage"
6. **Verify:** Results show Prefill Time, Decode Time, KV Cache Size, and Memory estimates
7. **Verify:** The token generation timeline chart renders correctly

### MAPE Target After All Phases

| Metric | Current | Target |
|--------|---------|--------|
| LLM MAPE (in-distribution) | 11.5% | < 15% |
| LLM MAPE (cross-GPU, after Colab data) | Unknown | < 25% |
| CNN MAPE | ~14% | No regression |

---

## Execution Order

| # | Phase | Effort | Dependency |
|---|-------|--------|------------|
| 1 | Fix gpu_predictor.py plumbing | 30 min | None |
| 2 | Fix dashboard.py LLM UI | 45 min | Phase 1 |
| 3 | Fix prediction_api.py | 20 min | Phase 1 |
| 4 | Separate prefill/decode models | 40 min | None |
| 5 | Synthetic scaling data | 30 min | None |
| 6 | Colab GPU collector script | 30 min | None |
| 7 | Hardware recommender | 45 min | Phase 4 |
| 8 | Test suite | 30 min | Phases 1-4 |
| 9 | Retrain on expanded data | 5 min | Phases 5-6 |
