# bench_v13_refactor

`bench_v12_hotfix15_py` Equivalent refactor of bench_v12_hotfix15_py — functionality and output remain 100% unchanged. The 30 nearly-identical inference scripts are consolidated into a common library + a small set of backends + a single CLI entry point.

---

## 1. Project Layout

```
bench_v13_refactor/
├── omm_bench/                       # Common library
│   ├── core.py                      # env + prompt + io_utils + runner
│   │                                #   · apply_env()
│   │                                #   · QUESTION_TEMPLATE / extract_answer / get_instruction
│   │                                #   · save_batch / resume_ids / build_img_map / get_meta
│   │                                #   · run_benchmark()
│   └── backends.py                  # 6 backend classes merged into one module
│                                    #   · LmdeployBackend         → 11 个 lmdeploy scripts
│                                    #   · LlavaNextBackend        → LLaVA-v1.6（± --no-bos）
│                                    #   · MllamaBackend           → Llama-3.2-11B-Vision（± --no-bos）
│                                    #   · LlavaOneVisionBackend   → LLaVA-OneVision-7B
│                                    #   · JanusBackend            → Janus-Pro-7B
│                                    #   · QwenVLMaxAPIBackend     → DashScope（--api-mode）
│                                    #   · BACKEND_REGISTRY        → name  → class lookup table
├── run_bench.py                     # Single entry point
├── configs/models.yaml              # Default parameters per model
├── eval_accuracy.py                 # Kept as-is
├── eval_accuracy_v22.py             # Kept as-is
├── acc_v1.py                        # Kept as-is
├── download_models.py               # Kept as-is
└── test_run.py                      # Kept as-is
```

---

## 2. Dependencies

```bash
pip install pyyaml pillow tqdm
# As needed:
pip install lmdeploy                 # lmdeploy backends
pip install transformers torch       # HF backends
pip install dashscope                # DashScope legacy / fixed
pip install openai                   # DashScope openai-compatible endpoint
```

---

## 3. CLI Sheet (one-to-one with the original scripts)

| Original script                   | New command |
|---|---|
| `Qwen2_5_VL_7B_Instruct_v12.py`   | `python run_bench.py --model qwen2_5_vl_7b` |
| `Qwen2_5_VL_72B_Instruct_v12.py`  | `python run_bench.py --model qwen2_5_vl_72b` |
| `Qwen3_VL_8B_Instruct_v12.py`     | `python run_bench.py --model qwen3_vl_8b` |
| `Qwen3_VL_32B_Instruct_v12.py`    | `python run_bench.py --model qwen3_vl_32b` |
| `InternVL2_5_8B_v12.py`           | `python run_bench.py --model internvl2_5_8b` |
| `InternVL2_5_38B_v12.py`          | `python run_bench.py --model internvl2_5_38b` |
| `InternVL2_5_78B_v12.py`          | `python run_bench.py --model internvl2_5_78b` |
| `InternVL3_5_8B_v12.py`           | `python run_bench.py --model internvl3_5_8b` |
| `InternVL3_5_38B_v12.py`          | `python run_bench.py --model internvl3_5_38b` |
| `MiniCPM_V_2_5_v12.py`            | `python run_bench.py --model minicpm_v_2_5` |
| `LLaVA_OneVision_7B_v12.py`       | `python run_bench.py --model llava_onevision_7b` |
| `LLaVA_v1_6_7B_v12.py`            | `python run_bench.py --model llava_v1_6_7b` |
| `LLaVA_v1_6_7B_v12_hotfix18.py`   | `python run_bench.py --model llava_v1_6_7b --no-bos` |
| `Llama_3_2_11B_Vision_Instruct_v12.py`         | `python run_bench.py --model llama_3_2_11b_vision` |
| `Llama_3_2_11B_Vision_Instruct_v12_hotfix18.py`| `python run_bench.py --model llama_3_2_11b_vision --no-bos` |
| `Janus_Pro_7B_v12.py`             | `python run_bench.py --model janus_pro_7b` |
| `Qwen_VL_Max_API_v12.py`          | `python run_bench.py --model qwen_vl_max_api --api-mode legacy` |
| `Qwen_VL_Max_API_v12_hotfix18.py` | `python run_bench.py --model qwen_vl_max_api --api-mode fixed` |
| OpenAI-compatible endpoint        | `python run_bench.py --model qwen_vl_max_api --api-mode openai` |

### Common CLI overrides (no YAML edit required)

| Argument         | Description |
|---|---|
| `--model-path`   | Override the model path |
| `--output`       | Override the JSONL output path |
| `--data-path`    |Override the QA JSON path (default /root/autodl-fs/OMM-Bench_Positive_QA_v1_8_test.json) |
| `--image-dir`    | Override the image directory (default /root/autodl-fs/ReplicaPano_test) |
| `--batch-size`   | Override batch size |
| `--tp`           | lmdeploy tensor-parallel |
| `--cuda-visible` | `CUDA_VISIBLE_DEVICES`（API `""`） |
| `--torch-dtype`  | HF backends dtype |
| `--add-bos / --no-bos` | LLaVA-Next / Whether LLaVA-Next / Mllama prepends BOS (aligned with v12 vs hotfix18) |
| `--api-mode`     | DashScope: `legacy / fixed / openai` |
| `--api-key`      | Or set via the DASHSCOPE_API_KEY environment variable |
| `--max-img-size` | API image upper bound |
| `--max-new-tokens` | Generation length (default 512) |

Examples:

```bash
# 78B with custom tp and output path
python run_bench.py --model internvl2_5_78b --tp 8 \
    --output /root/autodl-tmp/InternVL2_5_78B_v12_run2.json

# API version with a different image directory
export DASHSCOPE_API_KEY=sk-xxxx
python run_bench.py --model qwen_vl_max_api --api-mode fixed \
    --image-dir /root/other_image_root
```

---

## 4. Equivalence Guarantees

QUESTION_TEMPLATE, extract_answer, and get_instruction are byte-for-byte identical to the originals, now located in omm_bench/core.py.
The environment-variable combination (QWEN2_5_VL_MAX_PIXELS=602112 / QWEN_VL_MAX_PIXELS=602112 / QWEN2_5_VL_MIN_PIXELS=3136 / NCCL_P2P_DISABLE=1 / INTERNVL_MAX_NUM=12) is uniformly injected by apply_env() at the start of every run.
JSONL field order per line matches the original scripts: id → model_response → full_response → gt_answer → question → eval_mode → decoding → matched_image → task_type → task_type_detail → instruction_type.
resume_ids() logic is unchanged; resumption still relies on the set of ids already written to the JSONL file.
Error prefix conventions: lmdeploy / LLaVA-OneVision use INFER_ERROR: ...; LLaVA-Next / Mllama / Janus / Qwen-VL-Max use ERROR: ...; DashScope writes API_ERROR: ... after retries are exhausted. All are aligned with the originals.
eval_mode / decoding / desc retain their original values — see configs/models.yaml.

---

## 5. Maintenance Workflow

- **Change default parameters for a single model → edit the corresponding entry in configs/models.yaml.
- **Add a new model** add a new key to the YAML file; if no matching backend exists, add a new class in omm_bench/backends.py and register it in BACKEND_REGISTRY.
- **One-off override for the current run** → pass the appropriate --xxx argument on the command line; no YAML edit needed.
