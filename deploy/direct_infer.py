# -*- coding: utf-8 -*-
"""皮肤VQA提问接口：接收图片+问题，返回模型诊断结果。

启动方式（在 AutoDL 4090 实例上）：
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

# 模型路径支持环境变量覆盖；默认指向相对目录，部署时按实际路径设置
BASE = os.environ.get("MODEL_DIR", "models/Qwen2.5-VL-7B-Instruct")
ADAPTER = os.environ.get("ADAPTER_DIR", "models/adapters/rank16")

DX_CN2CODE = {
    "黑色素细胞痣": "nv", "黑素细胞痣": "nv",
    "黑色素瘤": "mel", "黑素瘤": "mel",
    "良性角化病": "bkl", "脂溢性角化": "bkl",
    "基底细胞癌": "bcc",
    "光化性角化": "akiec", "日光性角化": "akiec",
    "血管病变": "vasc", "血管性皮损": "vasc", "血管瘤": "vasc",
    "皮肤纤维瘤": "df",
}
DX_FALLBACK = [
    ("mel", ["黑色素瘤", "melanoma", "黑素瘤"]),
    ("nv", ["黑色素细胞痣", "melanocytic nevus", "色素痣", "nevus", "痣"]),
    ("akiec", ["光化性角化", "日光性角化", "actinic keratosis", "intraepithelial"]),
    ("bkl", ["良性角化", "benign keratosis", "seborrheic keratosis", "脂溢性角化", "老年斑"]),
    ("bcc", ["基底细胞癌", "basal cell carcinoma"]),
    ("df", ["皮肤纤维瘤", "dermatofibroma", "纤维瘤"]),
    ("vasc", ["血管病变", "血管性皮损", "血管瘤", "vascular lesion", "血管"]),
]
DX_CODE2CN = {
    "nv": "黑色素细胞痣",
    "mel": "黑色素瘤",
    "bkl": "良性角化病",
    "bcc": "基底细胞癌",
    "akiec": "光化性角化病",
    "vasc": "血管病变",
    "df": "皮肤纤维瘤",
}

# 与训练/评估基线提问口径一致（test_set.json 同款问法）。
# 消融实验证明：注入鉴别特征提示会使 micro 71.88%→31.68%（mel先验偏置），
# 故生产默认不启用，仅保留在 eval.py --hint 作消融开关（技术报告 5.5 节）。
DEFAULT_QUESTION = "基于这张图片，请分析可能的诊断结果和建议。"

model = None
processor = None


@asynccontextmanager
async def lifespan(app):
    global model, processor
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=768 * 768)
    model = VLModel.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def extract_dx(text: str) -> str:
    m = re.search(r"诊断为[：:]\s*([^\n。]+)", text)
    if m:
        seg = m.group(1)
        for cn, code in DX_CN2CODE.items():
            if cn in seg:
                return code
    for code, kws in DX_FALLBACK:
        for kw in kws:
            if kw.lower() in text.lower():
                return code
    return "unknown"


def infer(img_path: str, question: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=256, do_sample=False)
    pred = processor.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return pred.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(DEFAULT_QUESTION),
):
    img_bytes = await file.read()
    tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
    tmp.write_bytes(img_bytes)
    t0 = time.time()
    pred = infer(str(tmp), question)
    dx = extract_dx(pred)
    tmp.unlink(missing_ok=True)
    return {
        "answer": pred,
        "diagnosis": DX_CODE2CN.get(dx, "无法识别"),
        "code": dx,
        "latency_s": round(time.time() - t0, 2),
    }