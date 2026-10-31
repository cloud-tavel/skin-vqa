# -*- coding: utf-8 -*-
"""皮肤VQA后端（vLLM 代理版）：接收图片+问题，转发本机 vLLM 推理，返回结构化诊断。
小程序端接口完全不变（/health、/ask，返回字段一致）。

前置：vLLM 已在 127.0.0.1:8000 运行（--served-model-name skin-vqa）
启动：uvicorn main:app --host 0.0.0.0 --port 6006
"""
import base64
import json
import os
import re
import time

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# vLLM 服务地址与模型名支持环境变量覆盖（默认本机 8000 端口）
VLLM_CHAT = os.environ.get("VLLM_CHAT", "http://127.0.0.1:8000/v1/chat/completions")
VLLM_MODELS = os.environ.get("VLLM_MODELS", "http://127.0.0.1:8000/v1/models")
MODEL = os.environ.get("VLLM_MODEL_NAME", "skin-vqa")

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
    "nv": "黑色素细胞痣", "mel": "黑色素瘤", "bkl": "良性角化病",
    "bcc": "基底细胞癌", "akiec": "光化性角化病", "vasc": "血管病变",
    "df": "皮肤纤维瘤",
}

# 与训练/评估基线提问口径一致（test_set.json 同款问法）。
# 消融实验证明：注入鉴别特征提示会使 micro 71.88%→31.68%，故不启用。
DEFAULT_QUESTION = "基于这张图片，请分析可能的诊断结果和建议。"

app = FastAPI()
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


@app.get("/health")
def health():
    try:
        up = requests.get(VLLM_MODELS, timeout=3).ok
    except Exception:
        up = False
    return {"status": "ok", "vllm": "up" if up else "down"}


@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(DEFAULT_QUESTION),
    history: str = Form(""),
):
    b64 = base64.b64encode(await file.read()).decode()
    mime = file.content_type or "image/jpeg"

    prior_messages = []
    if history:
        try:
            parsed = json.loads(history)
            for m in parsed[-8:]:
                role = m.get("role")
                content = str(m.get("content", "")).strip()
                if role in ("user", "assistant") and content:
                    prior_messages.append({"role": role, "content": [
                        {"type": "text", "text": content}
                    ]})
        except (ValueError, AttributeError):
            prior_messages = []

    payload = {
        "model": MODEL,
        "messages": prior_messages + [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": question},
        ]}],
        "max_tokens": 220,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(VLLM_CHAT, json=payload, timeout=180)
    r.raise_for_status()
    pred = r.json()["choices"][0]["message"]["content"].strip()
    dx = extract_dx(pred)
    return {
        "answer": pred,
        "diagnosis": DX_CODE2CN.get(dx, "无法识别"),
        "code": dx,
        "latency_s": round(time.time() - t0, 2),
        "turns": len(prior_messages) // 2 + 1,
    }


def tencent_sentence_recognition(audio: bytes) -> str:
    """调用腾讯云「一句话识别」转写 mp3 音频（延迟导入 SDK，未安装时不影响 /ask）。"""
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.asr.v20190614 import asr_client, models

    secret_id = os.environ.get("TENCENT_SECRET_ID", "").strip()
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "").strip()
    if not secret_id or not secret_key:
        raise RuntimeError("后端未配置腾讯云密钥（TENCENT_SECRET_ID / TENCENT_SECRET_KEY）")

    cred = credential.Credential(secret_id, secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "asr.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = asr_client.AsrClient(cred, "", client_profile)

    req = models.SentenceRecognitionRequest()
    params = {
        "ProjectId": 0,
        "SubServiceType": 2,          # 一句话识别
        "EngSerViceType": "16k_zh",   # 16k 中文普通话
        "SourceType": 1,              # 1 = POST 上传二进制音频
        "VoiceFormat": "mp3",
        "UsrAudioKey": str(int(time.time() * 1000)),
        "Data": base64.b64encode(audio).decode(),
        "DataLen": len(audio),
    }
    req.from_json_string(json.dumps(params))
    resp = client.SentenceRecognition(req)
    return resp.Result or ""


@app.post("/asr")
async def asr_transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    if not audio:
        return {"text": "", "error": "空音频"}
    if len(audio) > 3 * 1024 * 1024:
        return {"text": "", "error": "音频超过 3MB"}
    try:
        text = tencent_sentence_recognition(audio)
        return {"text": text, "error": ""}
    except Exception as e:
        return {"text": "", "error": "识别失败: " + str(e)}
