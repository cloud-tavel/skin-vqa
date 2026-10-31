# -*- coding: utf-8 -*-
"""把 HAM10000 原始数据转换成训练用的 conversations 格式。

输入：
    HAM10000_metadata.csv   （列：lesion_id, image_id, dx, dx_type, age, sex, localization）
    images/                 （ISIC_xxxxxxx.jpg）

输出（默认 skin_vqa_train.json）：
    [
      {
        "id": "0",
        "image": ["images/ISIC_0024306.jpg"],
        "conversations": [
          {"from": "user", "value": "<image>这是什么皮肤病？需要去医院吗？"},
          {"from": "assistant", "value": "根据图像特征分析，该皮损倾向于诊断为：黑色素细胞痣 (Melanocytic nevi)。\\n\\n【特征描述】：……\\n【医疗建议】：……"}
        ]
      }
    ]

约定（下游脚本依赖）：
  - 单图；conversations 恰好两轮（user / assistant）；
  - 回答中固定出现「诊断为：<中文病名> (<English>)」，train_balanced.py 与
    split_train_test.py 据此解析类别标签，改动措辞需同步核对两处正则。

用法：
    python build_dataset.py \
        --meta HAM10000_metadata.csv \
        --image_dir images \
        --out skin_vqa_train.json \
        --seed 42

生成后再做病灶级互斥划分：
    python split_train_test.py --data skin_vqa_train.json --meta HAM10000_metadata.csv \
        --image_dir images --out split --ratio 0.8 --seed 42
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

# dx 代码 → (中文名, 英文名, 特征描述, 医疗建议)
DX_INFO = {
    "akiec": (
        "光化性角化病", "Actinic keratoses",
        "皮损表现为红色、粗糙的鳞屑性斑块，通常发生在长期暴晒部位（如面部、手背）。",
        "这是皮肤癌（鳞状细胞癌）的癌前病变。建议进行冷冻治疗或外用药物治疗，并严格做好防晒措施。",
    ),
    "bcc": (
        "基底细胞癌", "Basal cell carcinoma",
        "常见表现为珍珠样的半透明结节，表面可见扩张的毛细血管，中央可能出现溃疡。",
        "这是一种常见的皮肤恶性肿瘤，但转移风险较低。建议手术切除，预后通常良好。",
    ),
    "bkl": (
        "良性角化病", "Benign keratosis-like lesions",
        "皮损通常像贴在皮肤上的蜡状斑块，表面粗糙，可能呈现疣状外观，常见于脂溢性角化病（老年斑）。",
        "这通常是良性病变，主要影响美观。如无不适可观察；如伴有瘙痒或为了美观，可考虑冷冻或激光治疗。",
    ),
    "df": (
        "皮肤纤维瘤", "Dermatofibroma",
        "皮损为坚硬的结节，通常为褐色或红褐色，捏起时皮肤可能出现凹陷（酒窝征）。",
        "这是良性的皮肤软组织肿瘤。一般无需处理，若有疼痛或持续增大，可考虑手术切除。",
    ),
    "mel": (
        "黑色素瘤", "Melanoma",
        "皮损表现为不对称的色素沉着，边界不规则，颜色深浅不一（可能有黑色、褐色、红色混合）。",
        "【高危预警】这是一种恶性程度较高的皮肤肿瘤。强烈建议您立即前往正规医院皮肤科进行皮肤镜检查或病理活检，以免延误治疗。",
    ),
    "nv": (
        "黑色素细胞痣", "Melanocytic nevi",
        "皮损通常呈现为均匀的褐色或棕色斑点，边界清晰，结构对称。",
        "这是常见的良性痣，一般不需要治疗。建议定期观察，如果出现大小、颜色或形状的突然变化，请及时复诊。",
    ),
    "vasc": (
        "血管病变", "Vascular lesions",
        "皮损呈现红色或紫色，按压可能褪色，通常由血管扩张或增生引起（如樱桃样血管瘤）。",
        "通常为良性血管增生。若无出血或疼痛，通常无需治疗；若影响美观，可选择激光治疗。",
    ),
}

# 问题模板：同一类别不同措辞，降低模型对单一问法的过拟合
QUESTIONS = [
    "基于这张图片，请分析可能的诊断结果和建议。",
    "请详细描述这张皮肤镜图像的特征并给出诊断。",
    "我的皮肤上长了这个东西，请问是什么？严重吗？",
    "医生，请帮我看一下这张皮肤照片，这是什么病？",
    "这个斑块看起来有点担心，能告诉我这是什么吗？",
    "这是什么皮肤病？需要去医院吗？",
]


def build_answer(dx: str) -> str:
    cn, en, desc, advice = DX_INFO[dx]
    return (
        f"根据图像特征分析，该皮损倾向于诊断为：{cn} ({en})。\n\n"
        f"【特征描述】：{desc}\n"
        f"【医疗建议】：{advice}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="HAM10000_metadata.csv", help="HAM10000 标注 CSV")
    ap.add_argument("--image_dir", default="images", help="图像目录（ISIC_xxx.jpg）")
    ap.add_argument("--out", default="skin_vqa_train.json", help="输出 JSON 路径")
    ap.add_argument("--seed", type=int, default=42, help="问题模板抽样种子（可复现）")
    args = ap.parse_args()

    meta_path, img_dir = Path(args.meta), Path(args.image_dir)
    if not meta_path.exists():
        raise SystemExit(f"找不到标注文件: {meta_path}")
    if not img_dir.is_dir():
        raise SystemExit(f"找不到图像目录: {img_dir}")

    rng = random.Random(args.seed)
    records, skipped, dist = [], Counter(), Counter()
    seen = set()

    with meta_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id, dx = row.get("image_id", "").strip(), row.get("dx", "").strip()
            if not image_id or dx not in DX_INFO:
                skipped["标注缺失或类别未知"] += 1
                continue
            if image_id in seen:
                skipped["重复 image_id"] += 1
                continue
            if not (img_dir / f"{image_id}.jpg").exists():
                skipped["本地缺图片"] += 1
                continue

            seen.add(image_id)
            dist[dx] += 1
            records.append({
                "id": str(len(records)),
                "image": [f"images/{image_id}.jpg"],
                "conversations": [
                    {"from": "user", "value": "<image>" + rng.choice(QUESTIONS)},
                    {"from": "assistant", "value": build_answer(dx)},
                ],
            })

    if not records:
        raise SystemExit("没有可用样本：请检查 --meta 与 --image_dir 是否对应")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"写入 {out_path}：{len(records)} 条样本")
    for dx in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {dx:<6} {dist[dx]:>5}")
    if skipped:
        print(f"跳过: {dict(skipped)}")


if __name__ == "__main__":
    main()
