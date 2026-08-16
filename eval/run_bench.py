"""Unified CLI entry point for all bench_v12 model scripts.

Usage
-----
    python run_bench.py --model qwen2_5_vl_7b
    python run_bench.py --model internvl2_5_78b --tp 8 --output out.json
    python run_bench.py --model llava_v1_6_7b --no-bos       # = hotfix18
    python run_bench.py --model qwen_vl_max_api --api-mode fixed
"""
import argparse
import os
import sys
import yaml

from omm_bench.core     import apply_env, run_benchmark
from omm_bench.backends import BACKEND_REGISTRY


def load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser(description="OMM-Bench unified runner")
    ap.add_argument('--model', required=True,
                    help='key in configs/models.yaml (e.g. qwen2_5_vl_7b)')
    ap.add_argument('--config', default='configs/models.yaml')
    # Generic overrides — no need to edit YAML for one-off runs
    ap.add_argument('--model-path')
    ap.add_argument('--data-path',
                    default='/root/autodl-fs/OMM-Bench_Positive_QA_v1_8_test.json')
    ap.add_argument('--image-dir',
                    default='/root/autodl-fs/ReplicaPano_test')
    ap.add_argument('--output', help='override output_file')
    ap.add_argument('--batch-size',     type=int)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--tp',             type=int,
                    help='lmdeploy tensor parallel size')
    ap.add_argument('--cuda-visible', default=None,
                    help='CUDA_VISIBLE_DEVICES (pass "" for API-only backends)')
    # HF-native backends
    ap.add_argument('--torch-dtype')
    ap.add_argument('--add-bos', dest='add_bos', action='store_true', default=None)
    ap.add_argument('--no-bos',  dest='add_bos', action='store_false')
    # API backend
    ap.add_argument('--api-mode', choices=['legacy', 'fixed', 'openai'])
    ap.add_argument('--api-key',      default=None)
    ap.add_argument('--max-img-size', type=int)
    return ap.parse_args()


def _pick(cfg, key, cli_val, default=None):
    """CLI value wins over YAML value; fall back to `default`."""
    return cli_val if cli_val is not None else cfg.get(key, default)


def main():
    args = parse_args()

    if not os.path.isfile(args.config):
        print(f"❌ config not found: {args.config}"); sys.exit(1)
    cfg_all = load_yaml(args.config) or {}
    cfg = cfg_all.get(args.model)
    if cfg is None:
        print(f"❌ model key '{args.model}' not in {args.config}")
        print(f"   available keys: {sorted(cfg_all.keys())}")
        sys.exit(1)

    # ── Environment setup ────────────────────────────────────────────
    cuda = (args.cuda_visible if args.cuda_visible is not None
            else cfg.get('cuda_visible', '0'))
    apply_env(cuda_visible=str(cuda))

    backend_name = cfg['backend']
    BackendCls   = BACKEND_REGISTRY[backend_name]

    # ── Backend-specific kwargs ──────────────────────────────────────
    b_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if backend_name == 'lmdeploy_pipe':
        b_kwargs['model_path'] = _pick(cfg, 'model_path', args.model_path)
        b_kwargs['tp']         = _pick(cfg, 'tp', args.tp, 1)
    elif backend_name in ('llava_next', 'mllama'):
        b_kwargs['model_path']  = _pick(cfg, 'model_path', args.model_path)
        b_kwargs['torch_dtype'] = _pick(cfg, 'torch_dtype', args.torch_dtype,
                                        'bfloat16')
        add_bos = (args.add_bos if args.add_bos is not None
                   else cfg.get('add_bos', True))
        b_kwargs['add_bos'] = add_bos
    elif backend_name == 'llava_onevision_hf':
        b_kwargs['model_path']  = _pick(cfg, 'model_path', args.model_path)
        b_kwargs['torch_dtype'] = _pick(cfg, 'torch_dtype', args.torch_dtype,
                                        'bfloat16')
    elif backend_name == 'janus':
        b_kwargs['model_path'] = _pick(cfg, 'model_path', args.model_path)
        b_kwargs['janus_root'] = cfg.get('janus_root',
                                         '/root/autodl-tmp/Janus')
    elif backend_name == 'qwen_vl_max_api':
        b_kwargs.update(
            api_key      = args.api_key or os.environ.get(
                              "DASHSCOPE_API_KEY", cfg.get('api_key', '')),
            api_mode     = _pick(cfg, 'api_mode',     args.api_mode,    'fixed'),
            max_img_size = _pick(cfg, 'max_img_size', args.max_img_size, 1024),
        )
    else:
        print(f"❌ unknown backend: {backend_name}"); sys.exit(1)

    backend = BackendCls(**b_kwargs)

    # ── Run ─────────────────────────────────────────────────────────
    run_benchmark(
        backend     = backend,
        data_path   = args.data_path,
        image_dir   = args.image_dir,
        output_file = _pick(cfg, 'output_file', args.output),
        eval_mode   = cfg['eval_mode'],
        decoding    = cfg['decoding'],
        desc        = cfg.get('desc', cfg['eval_mode']),
        batch_size  = _pick(cfg, 'batch_size', args.batch_size, 1),
        need_pil    = cfg.get('need_pil',
                              backend_name not in ('lmdeploy_pipe',)),
    )


if __name__ == '__main__':
    main()
