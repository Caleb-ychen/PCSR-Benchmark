"""OMM-Bench backends — all 6 model backends merged into one module.

Merged from:
  - backends/lmdeploy_pipe.py      → LmdeployBackend
  - backends/llava_next.py         → LlavaNextBackend
  - backends/mllama.py             → MllamaBackend
  - backends/llava_onevision_hf.py → LlavaOneVisionBackend
  - backends/janus.py              → JanusBackend
  - backends/qwen_vl_max_api.py    → QwenVLMaxAPIBackend

Each backend keeps its heavy dependencies (torch, transformers.*, lmdeploy,
dashscope, janus, openai) as LOCAL imports inside .load() / .infer() /
._call_*(), so importing this module does NOT pull in those libraries; only
the backend the user actually runs will trigger them.
"""
# ============================================================
# SECTION 1 — Shared standard-library / lightweight imports
# ============================================================
import os
import sys
import io
import base64
import time
import tempfile
from PIL import Image


# ============================================================
# SECTION 2 — LmdeployBackend  (11 lmdeploy models)
# ============================================================
class LmdeployBackend:
    """lmdeploy pipeline in TurboMind mode + hard-greedy GenerationConfig.

    Covers 11 originally-copy-pasted scripts:
      - Qwen2.5-VL-7B / -72B
      - Qwen3-VL-8B / -32B
      - InternVL2.5-8B / -38B / -78B
      - InternVL3.5-8B / -38B
      - MiniCPM-V-2.5
    Only differences between those scripts are model_path and tp — both are
    constructor parameters here, so one class replaces all of them.
    """
    error_prefix = 'INFER_ERROR'

    def __init__(self, model_path, *, tp=1, session_len=32768,
                 cache_max_entry_count=0.4, max_new_tokens=512):
        self.model_path = model_path
        self.tp = tp
        self.session_len = session_len
        self.cache = cache_max_entry_count
        self.max_new_tokens = max_new_tokens
        self.pipe = None
        self.gen_cfg = None
        self._load_image = None

    def load(self):
        from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        from lmdeploy.vl import load_image as _load_img

        engine_cfg = TurbomindEngineConfig(
            tp=self.tp,
            session_len=self.session_len,
            cache_max_entry_count=self.cache,
        )
        self.pipe = pipeline(self.model_path, backend_config=engine_cfg)
        self.gen_cfg = GenerationConfig(
            do_sample=False, top_k=1, top_p=1.0,
            temperature=1.0, max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.0,
        )
        self._load_image = _load_img

    def infer(self, instruction, m):
        # lmdeploy tuple mode: (text, image) — vision tokens auto-injected
        resp = self.pipe((instruction, self._load_image(m['img_path'])),
                         gen_config=self.gen_cfg)
        return resp.text if hasattr(resp, 'text') else str(resp)


# ============================================================
# SECTION 3 — LlavaNextBackend  (LLaVA-v1.6, v12 + hotfix18)
# ============================================================
class LlavaNextBackend:
    """LLaVA-v1.6 (LLaVA-Next) HuggingFace transformers backend.

    Covers both `LLaVA_v1_6_7B_v12.py` (add_bos=True) and
    `LLaVA_v1_6_7B_v12_hotfix18.py` (add_bos=False) — the only difference
    between those two originals is whether the BOS token is prepended.
    """
    error_prefix = 'ERROR'

    def __init__(self, model_path, *, torch_dtype='float16',
                 device_map='auto', max_new_tokens=512, add_bos=True):
        self.model_path     = model_path
        self.torch_dtype    = torch_dtype
        self.device_map     = device_map
        self.max_new_tokens = max_new_tokens
        self.add_bos        = add_bos

    def load(self):
        import torch
        from transformers import (LlavaNextProcessor,
                                  LlavaNextForConditionalGeneration)
        dtype = getattr(torch, self.torch_dtype)
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=self.device_map
        ).eval()

    def infer(self, instruction, m):
        import torch
        image = m['pil_image'].convert('RGB')
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        if self.add_bos:
            bos = self.processor.tokenizer.bos_token or ""
            prompt = bos + prompt
        inputs = self.processor(images=image, text=prompt,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=1.0, top_k=1,
                repetition_penalty=1.0,
            )
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_tokens, skip_special_tokens=True)


# ============================================================
# SECTION 4 — MllamaBackend  (Llama-3.2-11B-Vision, v12 + hotfix18)
# ============================================================
class MllamaBackend:
    """Llama-3.2-11B-Vision HuggingFace transformers backend.

    Covers both `Llama_3_2_11B_Vision_Instruct_v12.py` (add_bos=True) and
    `Llama_3_2_11B_Vision_Instruct_v12_hotfix18.py` (add_bos=False) — the
    only difference is whether the BOS token is prepended.
    """
    error_prefix = 'ERROR'

    def __init__(self, model_path, *, torch_dtype='bfloat16',
                 device_map='auto', max_new_tokens=512, add_bos=True):
        self.model_path     = model_path
        self.torch_dtype    = torch_dtype
        self.device_map     = device_map
        self.max_new_tokens = max_new_tokens
        self.add_bos        = add_bos

    def load(self):
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        dtype = getattr(torch, self.torch_dtype)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=self.device_map
        )
        self.model.eval()

    def infer(self, instruction, m):
        import torch
        image = m['pil_image'].convert('RGB')
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        if self.add_bos:
            bos = (self.processor.tokenizer.bos_token
                   if hasattr(self.processor, "tokenizer") else "") or ""
            input_text = bos + input_text
        inputs = self.processor(image, input_text,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=1.0, top_k=1,
                repetition_penalty=1.0,
            )
        gen_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_tokens, skip_special_tokens=True)


# ============================================================
# SECTION 5 — LlavaOneVisionBackend  (LLaVA-OneVision-7B)
# ============================================================
class LlavaOneVisionBackend:
    """LLaVA-OneVision-7B HuggingFace-native backend."""
    error_prefix = 'INFER_ERROR'

    def __init__(self, model_path, *, torch_dtype='bfloat16',
                 device_map='auto', max_new_tokens=512):
        self.model_path     = model_path
        self.torch_dtype    = torch_dtype
        self.device_map     = device_map
        self.max_new_tokens = max_new_tokens

    def load(self):
        import torch
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                  AutoProcessor)
        dtype = getattr(torch, self.torch_dtype)
        print(f"🏗️ Loading Model from {self.model_path}...")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype,
            device_map=self.device_map, trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def infer(self, instruction, m):
        import torch
        raw_image = Image.open(m['img_path']).convert('RGB')
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n"
                  "<|im_start|>assistant\n")
        inputs = self.processor(images=raw_image, text=prompt,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.processor.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()


# ============================================================
# SECTION 6 — JanusBackend  (Janus-Pro-7B, with sys-path + transformers patch)
# ============================================================
class JanusBackend:
    """Janus-Pro-7B native backend.

    Preserves the two runtime patches the original script needs:
      1. sys.path.insert(0, '/root/autodl-tmp/Janus')
      2. Compatibility shim: `transformers.ImageProcessor = BaseImageProcessor`
    """
    error_prefix = 'ERROR'

    def __init__(self, model_path, *, janus_root='/root/autodl-tmp/Janus',
                 max_new_tokens=512):
        self.model_path     = model_path
        self.janus_root     = janus_root
        self.max_new_tokens = max_new_tokens

    def load(self):
        import torch
        # --- Janus 路径补丁 ---
        if self.janus_root and self.janus_root not in sys.path:
            sys.path.insert(0, self.janus_root)
        # --- Transformers 兼容性补丁 ---
        import transformers
        if not hasattr(transformers, 'ImageProcessor'):
            from transformers import BaseImageProcessor
            transformers.ImageProcessor = BaseImageProcessor

        from janus.models import MultiModalityCausalLM, VLChatProcessor
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt = MultiModalityCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(torch.bfloat16).cuda().eval()

    def infer(self, instruction, m):
        import torch
        pil_img = m['pil_image']
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_img.save(tmp.name)
            tmp_path = tmp.name
        try:
            conversation = [
                {"role": "<|User|>",
                 "content": f"<image_placeholder>\n{instruction}",
                 "images":  [tmp_path]},
                {"role": "<|Assistant|>", "content": ""}
            ]
            prep = self.vl_chat_processor(
                conversations=conversation, images=[pil_img],
                force_batchify=True
            ).to(self.vl_gpt.device)
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prep)
            with torch.no_grad():
                outputs = self.vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prep.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False, use_cache=True,
                )
            return self.tokenizer.decode(outputs[0].cpu().tolist(),
                                         skip_special_tokens=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================
# SECTION 7 — QwenVLMaxAPIBackend  (DashScope, v12 legacy + hotfix18 fixed + openai)
# ============================================================
MODEL_NAME_DEFAULT = "qwen-vl-max"


class QwenVLMaxAPIBackend:
    """Qwen-VL-Max DashScope backend, unified for v12 + hotfix18.

    api_mode:
      legacy  → original v12: flat 2 s × 3 retry, no None-guard, no resize
      fixed   → hotfix18:      None-guard + exponential backoff + resize
      openai  → hotfix18 OpenAI-compat endpoint
    """
    # Matches the outer catch's 'ERROR' prefix used by the original v12 API script
    error_prefix = 'ERROR'

    def __init__(self, *, api_key=None, model_name=MODEL_NAME_DEFAULT,
                 api_mode='fixed', max_img_size=1024, max_new_tokens=512):
        self.api_key        = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model_name     = model_name
        self.api_mode       = api_mode
        self.max_img_size   = max_img_size
        self.max_new_tokens = max_new_tokens

    # ---------------- lifecycle ----------------
    def load(self):
        if not self.api_key or self.api_key.startswith("YOUR_API_KEY"):
            print("❌ DASHSCOPE_API_KEY missing or placeholder.")
            sys.exit(1)
        if self.api_mode in ('legacy', 'fixed'):
            try:
                import dashscope
                dashscope.api_key = self.api_key
            except ImportError:
                print("❌ dashscope not installed: pip install dashscope")
                sys.exit(1)

    # ---------------- helpers ----------------
    def _pil_to_b64(self, pil_img):
        w, h = pil_img.size
        if self.api_mode != 'legacy' and max(w, h) > self.max_img_size:
            s = self.max_img_size / max(w, h)
            pil_img = pil_img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.convert('RGB').save(
            buf, format='JPEG',
            quality=85 if self.api_mode != 'legacy' else 95,
        )
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # ---------------- inference ----------------
    def infer(self, instruction, m):
        img_b64 = self._pil_to_b64(m['pil_image'])
        if self.api_mode == 'openai':
            return self._call_openai_compat(img_b64, instruction)
        return self._call_dashscope(img_b64, instruction,
                                    guard_none=(self.api_mode == 'fixed'))

    # dashscope path (covers legacy + fixed)
    def _call_dashscope(self, img_b64, instruction, *, guard_none):
        from dashscope import MultiModalConversation
        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img_b64}"},
                {"text":  instruction},
            ],
        }]
        full_resp = ''
        for attempt in range(3):
            try:
                resp = MultiModalConversation.call(
                    model=self.model_name, messages=messages,
                    max_tokens=self.max_new_tokens,
                    top_k=1, temperature=1.0,
                )
                if guard_none:
                    if resp is None:
                        raise RuntimeError("DashScope returned None response object")
                    if resp.output is None:
                        code = getattr(resp, 'code',       'UNKNOWN_CODE')
                        msg  = getattr(resp, 'message',    'no message')
                        req  = getattr(resp, 'request_id', 'no request_id')
                        raise RuntimeError(
                            f"output=None | code={code} | message={msg} | request_id={req}")
                full_resp = resp.output.choices[0].message.content[0]["text"]
                break
            except Exception as e:
                wait = 2 ** (attempt + 1) if guard_none else 2
                print(f"  ⚠ Attempt {attempt + 1}/3 failed: {e}  (retry in {wait}s)")
                if attempt < 2:
                    time.sleep(wait)
                else:
                    full_resp = f'API_ERROR: {e}'
        return full_resp

    # openai-compat endpoint
    def _call_openai_compat(self, img_b64, instruction):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed: pip install openai")
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        full_resp = ''
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": instruction},
                        ],
                    }],
                    max_tokens=self.max_new_tokens,
                    temperature=1.0, top_p=1.0,
                )
                full_resp = response.choices[0].message.content
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"  ⚠ Attempt {attempt + 1}/3 failed: {e}  (retry in {wait}s)")
                if attempt < 2:
                    time.sleep(wait)
                else:
                    full_resp = f'API_ERROR: {e}'
        return full_resp


# ============================================================
# SECTION 8 — Registry (was backends/__init__.py)
# ============================================================
BACKEND_REGISTRY = {
    'lmdeploy_pipe':      LmdeployBackend,
    'llava_next':         LlavaNextBackend,
    'mllama':             MllamaBackend,
    'llava_onevision_hf': LlavaOneVisionBackend,
    'janus':              JanusBackend,
    'qwen_vl_max_api':    QwenVLMaxAPIBackend,
}
