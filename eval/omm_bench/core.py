"""OMM-Bench core: env + prompt + io_utils + runner merged into one module.

Merged from:
  - omm_bench/env.py       → apply_env()
  - omm_bench/prompt.py    → QUESTION_TEMPLATE / extract_answer() / get_instruction()
  - omm_bench/io_utils.py  → save_batch / resume_ids / build_img_map / get_meta
  - omm_bench/runner.py    → run_benchmark()

All function bodies are byte-for-byte identical to the original 4 files.
"""
# ============================================================
# SECTION 1 — Imports (union of the 4 original modules)
# ============================================================
import os
import re
import sys
import json
from datetime import datetime, timedelta
from PIL import Image

try:
    import tqdm as tqdm_module
except ImportError:
    tqdm_module = None


# ============================================================
# SECTION 2 — env.py :: apply_env
# ============================================================
def apply_env(cuda_visible: str = "0", disable_nccl_p2p: bool = True):
    """Apply the environment variables shared across every v12 inference script.

    - CUDA_VISIBLE_DEVICES: pass "" for API-only backends (no GPU).
    - NCCL_P2P_DISABLE: mirrors the "1" that lmdeploy scripts hard-code.
    - Unified pixel budget (602112) — balances visual detail and CoT room,
      essential for panoramic spatial tasks to avoid token overflow.
    - INTERNVL_MAX_NUM: only InternVL scripts set this, but harmless elsewhere.
    """
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if disable_nccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["QWEN2_5_VL_MAX_PIXELS"] = "602112"
    os.environ["QWEN_VL_MAX_PIXELS"]    = "602112"
    os.environ["QWEN2_5_VL_MIN_PIXELS"] = "3136"
    os.environ.setdefault("INTERNVL_MAX_NUM", "12")


# ============================================================
# SECTION 3 — prompt.py :: QUESTION_TEMPLATE / extract_answer / get_instruction
# ============================================================
# Aligned with test_run.py: CoT prompt, no per-task branches.
# Combined instructions for standardized extraction and logic anchoring.
QUESTION_TEMPLATE = ("{question} \n"
    "Answer the question concisely. "
    "Output the thinking process in <think> </think> and "
    "the final answer (a single word or number) in <answer> </answer> tags.")


def extract_answer(text: str) -> str:
    """Strip <think>, extract <answer>...</answer>; fallback to first line[:200]."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    m = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()[:200]
    return text.strip().split('\n')[0][:200]


def get_instruction(item) -> str:
    """Return QUESTION_TEMPLATE filled with the question field.

    No per-task branches or suffix hints — aligned with test_run.py.
    """
    question = str(item.get('question_class', ''))
    return QUESTION_TEMPLATE.format(question=question)


# ============================================================
# SECTION 4 — io_utils.py :: save_batch / resume_ids / build_img_map / get_meta
# ============================================================
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')


def save_batch(output_file: str, new_results: list, total_written: int) -> int:
    """Append JSONL records; return updated total. tqdm-friendly print."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for r in new_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    total_written += len(new_results)
    msg = f"  ✓ Checkpoint: {total_written} records → {output_file}"
    if tqdm_module:
        tqdm_module.tqdm.write(msg)
    else:
        print(msg)
    return total_written


def resume_ids(output_file: str) -> set:
    """Read existing JSONL output and return the set of already-processed ids."""
    processed_ids = set()
    if os.path.isfile(output_file):
        with open(output_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if 'id' in r:
                        processed_ids.add(str(r['id']))
                except json.JSONDecodeError:
                    pass
        print(f"↩ Resumed: {len(processed_ids)} already processed.")
    else:
        # touch empty file so subsequent appends succeed
        open(output_file, 'w').close()
    return processed_ids


def build_img_map(image_dir: str) -> dict:
    """Walk IMAGE_DIR recursively, build {id_stem: full_path} map."""
    img_map = {}
    for root, _, files in os.walk(image_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(IMG_EXTS):
                stem = os.path.splitext(fname)[0]
            else:
                stem = fname
            img_map[stem] = fpath
    return img_map


def get_meta(item, img_map: dict, load_pil: bool = False) -> dict:
    """Extract metadata from a JSON QA record.

    load_pil=True → attempt PIL.Image.open (needed by HF transformers / API
    backends that consume PIL objects directly).
    """
    item_id  = str(item.get('id', 'UNKNOWN'))
    img_path = img_map.get(item_id)
    matched  = item_id
    pil_image = None

    if img_path and os.path.isfile(img_path):
        matched = os.path.basename(img_path)
        if load_pil:
            try:
                pil_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                pil_image = None
                matched   = f'LOAD_ERROR:{e}'
    else:
        img_path = None
        matched  = 'NOT_FOUND'

    return dict(
        item_id   = item_id,
        img_path  = img_path,
        pil_image = pil_image,
        matched   = matched,
        gt_answer        = str(item.get('answer_class',    '')),
        question         = str(item.get('question_class',  '')),
        task_type        = str(item.get('task_type',        '')),
        instruction_type = str(item.get('task_type_detail', item.get('task_type', ''))),
    )


# ============================================================
# SECTION 5 — runner.py :: run_benchmark
# ============================================================
def run_benchmark(*, backend, data_path, image_dir, output_file,
                  eval_mode, decoding, desc,
                  batch_size=1, need_pil=False):
    """Run inference for one model.

    Parameters
    ----------
    backend : object with two methods
        .load()                        → prepares model (called once)
        .infer(instruction, meta)      → returns raw text (full_resp)
        optional attribute .error_prefix ('INFER_ERROR' or 'ERROR') to match
        the exact string prefix used by the corresponding original v12 script.
    All output JSONL fields are identical to the original v12 scripts.
    """
    if not os.path.isfile(data_path):
        print(f"❌ DATA_PATH not found: {data_path}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"❌ IMAGE_DIR not found: {image_dir}")
        sys.exit(1)

    backend.load()

    with open(data_path, encoding='utf-8') as _f:
        data = json.load(_f)
    print(f"▶ Loaded {len(data)} items from {data_path}")

    img_map = build_img_map(image_dir)
    print(f"▶ Image map: {len(img_map)} files in {image_dir}")

    processed_ids = resume_ids(output_file)
    remaining     = [it for it in data if str(it.get('id', '')) not in processed_ids]
    print(f"▶ Remaining: {len(remaining)} / {len(data)} items")

    total_written = len(processed_ids)
    batch_buf     = []
    start_time    = datetime.now()

    tqdm_iter = tqdm_module.tqdm(remaining, desc=desc) if tqdm_module else remaining
    for item in tqdm_iter:
        m           = get_meta(item, img_map, load_pil=need_pil)
        instruction = get_instruction(item)

        # Uniform "no image" handling — same string every original script wrote
        if (need_pil and m['pil_image'] is None) or (not need_pil and m['img_path'] is None):
            full_resp = 'IMAGE_NONE'
        else:
            try:
                full_resp = backend.infer(instruction, m)
            except Exception as e:
                # Preserve original two error prefixes:
                # 'INFER_ERROR' for lmdeploy path; 'ERROR' for HF/native path.
                prefix = getattr(backend, 'error_prefix', 'INFER_ERROR')
                full_resp = f'{prefix}: {e}'

        record = {
            'id':               m['item_id'],
            'model_response':   extract_answer(full_resp),
            'full_response':    full_resp,
            'gt_answer':        m['gt_answer'],
            'question':         m['question'],
            'eval_mode':        eval_mode,
            'decoding':         decoding,
            'matched_image':    m['matched'],
            'task_type':        m['task_type'],
            'task_type_detail': m['instruction_type'],
            'instruction_type': m['instruction_type'],
        }
        batch_buf.append(record)
        if len(batch_buf) >= batch_size:
            total_written = save_batch(output_file, batch_buf, total_written)
            batch_buf = []

    if batch_buf:
        total_written = save_batch(output_file, batch_buf, total_written)

    elapsed = timedelta(seconds=int((datetime.now() - start_time).total_seconds()))
    print(f"\n✅ Done: {total_written} items | Output: {output_file} | Elapsed: {elapsed}")
    print(f"   Model: {eval_mode} | Decoding: {decoding} | Batch: {batch_size}")
