# -*- coding: utf-8 -*-
"""vLLM 延迟基准：取测试集前 N 张图逐条请求，统计平均/P50/最快/最慢耗时。"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests

# 数据根目录与 vLLM 地址支持环境变量覆盖
ROOT = Path(os.environ.get("DATA_ROOT", "."))
URL = os.environ.get("VLLM_CHAT", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL_NAME", "skin-vqa")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="测试样本数")
args = ap.parse_args()

samples = json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))[: args.n]
times = []
for i, s in enumerate(samples, 1):
    b64 = base64.b64encode((ROOT / s["image"]).read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": s["question"]},
        ]}],
        "max_tokens": 220,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=180)
    dt = time.time() - t0
    r.raise_for_status()
    ans = r.json()["choices"][0]["message"]["content"]
    times.append(dt)
    print(f"[{i}/{len(samples)}] {dt:.2f}s  gt={s['gt_dx']:6s}  {ans[:36]!r}", flush=True)

times.sort()
n = len(times)
print(f"\n样本: {n}  平均: {sum(times)/n:.2f}s  P50: {times[n//2]:.2f}s  "
      f"最快: {times[0]:.2f}s  最慢: {times[-1]:.2f}s")
