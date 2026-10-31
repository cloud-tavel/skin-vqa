# -*- coding: utf-8 -*-
"""皮肤VQA评估脚本：基座模型 + 可选LoRA适配器，在测试集上测诊断准确率。

在 upload_pkg 解压后的目录下运行：
    python eval.py --adapter adapters/rank16 --limit 20 --out smoke.csv   # 冒烟测试
    python eval.py --adapter adapters/rank16 --out results_rank16.csv
    python eval.py --adapter adapters/rank8  --out results_rank8.csv
    python eval.py --adapter none           --out results_base.csv
"""
import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel

# 微调后模型输出为"诊断为：XXX"格式，先按精确病名匹配
DX_CN2CODE = {
    "黑色素细胞痣": "nv", "黑素细胞痣": "nv",
    "黑色素瘤": "mel", "黑素瘤": "mel",
    "良性角化病": "bkl", "脂溢性角化": "bkl",
    "基底细胞癌": "bcc",
    "光化性角化": "akiec", "日光性角化": "akiec",
    "血管病变": "vasc", "血管性皮损": "vasc", "血管瘤": "vasc",
    "皮肤纤维瘤": "df",
}
# 兜底关键词（顺序即优先级，特异性强的在前；用于基座模型自由格式输出）
DX_FALLBACK = [
    ("mel", ["黑色素瘤", "melanoma", "黑素瘤"]),
    ("nv", ["黑色素细胞痣", "melanocytic nevus", "色素痣", "nevus", "痣"]),
    ("akiec", ["光化性角化", "日光性角化", "actinic keratosis", "intraepithelial"]),
    ("bkl", ["良性角化", "benign keratosis", "seborrheic keratosis", "脂溢性角化", "老年斑"]),
    ("bcc", ["基底细胞癌", "basal cell carcinoma"]),
    ("df", ["皮肤纤维瘤", "dermatofibroma", "纤维瘤"]),
    ("vasc", ["血管病变", "血管性皮损", "血管瘤", "vascular lesion", "血管"]),
]

# C1：鉴别特征提示词后缀（--hint 开启时追加在原始问题后，用于提升 nv/mel/df 等易混淆类别的判别）
DX_HINT = (
    " 鉴别时请重点观察皮损的不对称性、边界是否规则、颜色是否均匀，"
    "以及是否存在色素网络、伪足、蓝白幕等结构："
    "黑色素瘤常不对称、边界模糊、颜色不均；黑色素细胞痣通常对称、边界清晰、颜色均匀；"
    "皮肤纤维瘤多为坚硬隆起的褐色小结节。请以“诊断为：XXX”开头给出结论。"
)

FIELDS = ["id", "image", "question", "gt_dx", "pred_dx", "correct",
          "latency_s", "pred_text"]


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


def save_rows(rows, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, out_path: Path):
    per = defaultdict(lambda: [0, 0])
    unknown = sum(1 for r in rows if r["pred_dx"] == "unknown")
    correct = sum(r["correct"] for r in rows)
    for r in rows:
        per[r["gt_dx"]][1] += 1
        per[r["gt_dx"]][0] += r["correct"]
    n = len(rows)
    macro = sum(c / t for c, t in per.values()) / len(per) if per else 0
    avg_lat = sum(r["latency_s"] for r in rows) / n
    lines = [
        f"样本数: {n}",
        f"总体准确率(micro): {correct}/{n} = {correct / n:.4f}",
        f"各类平均准确率(macro): {macro:.4f}",
        f"无法解析诊断比例: {unknown}/{n} = {unknown / n:.4f}",
        f"平均生成耗时: {avg_lat:.2f}s/条",
        "",
        f"{'类别':<8}{'正确':>5}{'总数':>5}  准确率",
    ]
    for dx in sorted(per):
        c, t = per[dx]
        lines.append(f"{dx:<8}{c:>5}{t:>5}  {c / t:.3f}")
    text = "\n".join(lines)
    print("\n" + text)
    out_path.with_suffix(".summary.txt").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("MODEL_DIR", "models/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--adapter", default="none", help="adapter目录路径，或none表示纯基座")
    ap.add_argument("--test", default="test_set.json")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条，调试用")
    ap.add_argument("--hint", action="store_true", help="在问题后追加鉴别特征提示词（C1）")
    args = ap.parse_args()

    test_path = Path(args.test)
    samples = json.loads(test_path.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[:args.limit]
    print(f"测试样本: {len(samples)}  adapter: {args.adapter}")

    # 768*768 与训练时 image_max_pixels 保持一致
    processor = AutoProcessor.from_pretrained(args.base, max_pixels=768 * 768)
    model = VLModel.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    if args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = []
    t_start = time.time()
    for i, s in enumerate(samples):
        img_path = (test_path.parent / s["image"]).resolve()
        q = s["question"] + (DX_HINT if args.hint else "")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": q},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           return_tensors="pt").to(model.device)
        t_gen = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_dx = extract_dx(pred)
        rows.append({
            "id": s["id"], "image": s["image"], "question": q,
            "gt_dx": s["gt_dx"], "pred_dx": pred_dx,
            "correct": int(pred_dx == s["gt_dx"]),
            "latency_s": round(time.time() - t_gen, 2),
            "pred_text": pred.replace("\n", " ")[:500],
        })
        if (i + 1) % 20 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            eta = (time.time() - t_start) / (i + 1) * (len(samples) - i - 1)
            print(f"[{i + 1}/{len(samples)}] acc={acc:.3f} "
                  f"剩余约{eta / 60:.1f}分钟", flush=True)
            save_rows(rows, Path(args.out))  # 中途也保存，防止意外中断丢结果

    save_rows(rows, Path(args.out))
    summarize(rows, Path(args.out))
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
