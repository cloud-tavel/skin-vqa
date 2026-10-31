# -*- coding: utf-8 -*-
"""皮肤 VQA 严格互斥的 train/test 划分脚本（修复数据泄漏 S1）。

背景：原 make_test_set.py 直接从 skin_vqa_train.json 分层抽样得到 505 例"测试集"，
未从训练集中剔除，导致训练/评估同源（评估指标偏乐观、不能代表泛化能力）。

本脚本按 HAM10000 的 lesion_id 分组做严格互斥划分（同一病灶的多张影像必然同属一侧），
在病灶级别按类别分层 8:2 抽样，确保训练集与评估集在图片与病灶两级均零重叠。

样本口径与 make_test_set.py 一致：单图、本地图片存在、对话完整、答案含可解析诊断名。

用法：
    python split_train_test.py \
        --data /path/to/skin_vqa_train.json \
        --meta /path/to/HAM10000_metadata.csv \
        --image_dir /path/to/images \
        --out split --ratio 0.8 --seed 42

产物（写入 --out 目录）：
    skin_vqa_train_split.json   # LLaMA-Factory conversations 格式（新训练集，约 8 成）
    skin_vqa_test_split.json    # eval.py 格式 {id,image,question,gt_dx,gt_answer}（约 2 成）
    split_report.txt            # 各类别分布与重叠校验报告

服务器复评（图片与 json 按同布局上传后，在仓库根目录下）：
    1) 重训：bash run_retrain.sh train   （TRAIN_DATA 指向 split 后的训练集）
    2) 评估：python eval.py --test split/skin_vqa_test_split.json --adapter <adapter>
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DX_CN2CODE = {
    "黑色素细胞痣": "nv", "黑色素瘤": "mel", "良性角化病": "bkl",
    "基底细胞癌": "bcc", "光化性角化病": "akiec", "血管病变": "vasc",
    "皮肤纤维瘤": "df",
}


def gt_dx_of(answer: str):
    m = re.search(r"诊断为[：:]\s*(.+?)[。\n（(]", answer)
    if not m:
        return None
    name = m.group(1).strip()
    for cn, code in DX_CN2CODE.items():
        if cn in name:
            return code
    return None


def image_id_of(item):
    imgs = item.get("image") or []
    if not imgs:
        return None
    return Path(imgs[0]).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="skin_vqa_train.json")
    ap.add_argument("--meta", default="HAM10000_metadata.csv")
    ap.add_argument("--image_dir", default=None, help="图片目录；缺省取 data 同目录 images/")
    ap.add_argument("--out", default="split")
    ap.add_argument("--ratio", type=float, default=0.8, help="训练集占比，默认 0.8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    meta_path = Path(args.meta)
    img_dir = Path(args.image_dir) if args.image_dir else data_path.parent / "images"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) metadata：image_id -> (lesion_id, dx)
    meta = {}
    with meta_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["image_id"]] = (row["lesion_id"], row["dx"])

    # 2) 过滤可用样本（口径同 make_test_set.py）
    data = json.loads(data_path.read_text(encoding="utf-8"))
    pool, skipped = [], Counter()
    for item in data:
        imgs = item.get("image") or []
        if len(imgs) != 1:
            skipped["多图/无图"] += 1
            continue
        img_name = Path(imgs[0]).name
        if not (img_dir / img_name).exists():
            skipped["本地缺图片"] += 1
            continue
        conv = item.get("conversations") or []
        if len(conv) < 2:
            skipped["对话不完整"] += 1
            continue
        dx = gt_dx_of(conv[1]["value"])
        if dx is None:
            skipped["答案无诊断名"] += 1
            continue
        pool.append(item)

    print(f"数据总数: {len(data)}  可用: {len(pool)}  跳过: {dict(skipped)}")

    # 3) 按病灶（lesion_id）分组；dx 取 metadata 真值，缺 metadata 时用答案解析兜底
    lesion_groups = defaultdict(list)   # lesion_id -> [item]
    fallback = 0
    for item in pool:
        iid = image_id_of(item)
        mr = meta.get(iid)
        if mr is None:
            lesion_groups[f"img:{iid}"].append(item)  # 退化为按图成组，绝不与其它图混组
            fallback += 1
            continue
        lesion_groups[mr[0]].append(item)
    if fallback:
        print(f"警告：{fallback} 个样本无 metadata，按单图成组处理")

    def dx_of_group(items):
        for it in items:
            conv = it.get("conversations") or []
            if len(conv) >= 2:
                d = gt_dx_of(conv[1]["value"])
                if d:
                    return d
        return "unknown"

    by_dx = defaultdict(list)  # dx -> [(lesion_id, [item])]
    for lid, items in lesion_groups.items():
        by_dx[dx_of_group(items)].append((lid, items))

    # 4) 病灶级分层划分：目标训练侧图片占比 ≈ ratio
    random.seed(args.seed)
    train_items, test_items = [], []
    for dx in sorted(by_dx):
        groups = by_dx[dx]
        n_images = sum(len(it) for _, it in groups)
        random.shuffle(groups)
        target_train = max(1, round(n_images * args.ratio))
        cur = 0
        for g in groups:
            if cur < target_train:
                train_items.extend(g[1])
                cur += len(g[1])
            else:
                test_items.extend(g[1])

    random.shuffle(train_items)
    random.shuffle(test_items)

    # 5) 写新训练集（保持原 conversations 结构）
    (out_dir / "skin_vqa_train_split.json").write_text(
        json.dumps(train_items, ensure_ascii=False, indent=1), encoding="utf-8")

    # 6) 写 eval.py 格式测试集；image 为相对 out_dir 的路径（假定 images/ 与 out_dir 同级）
    test_list = []
    for i, it in enumerate(test_items):
        conv = it.get("conversations") or []
        question = conv[0]["value"].replace("<image>", "").strip() if conv else ""
        gt_answer = conv[1]["value"] if len(conv) >= 2 else ""
        img_name = Path((it.get("image") or [""])[0]).name
        test_list.append({"id": str(i), "image": f"../images/{img_name}",
                          "question": question,
                          "gt_dx": gt_dx_of(gt_answer) or "unknown",
                          "gt_answer": gt_answer})
    (out_dir / "skin_vqa_test_split.json").write_text(
        json.dumps(test_list, ensure_ascii=False, indent=1), encoding="utf-8")

    # 7) 报告 + 两级重叠校验
    tr_cnt = Counter()
    for it in train_items:
        conv = it.get("conversations") or []
        tr_cnt[gt_dx_of(conv[1]["value"]) if len(conv) >= 2 else "unknown"] += 1
    te_cnt = Counter(t["gt_dx"] for t in test_list)

    tr_img = {image_id_of(it) for it in train_items}
    te_img = {Path(t["image"]).stem for t in test_list}
    img_overlap = tr_img & te_img

    def lesion_of(iid):
        mr = meta.get(iid)
        return mr[0] if mr else f"img:{iid}"

    tr_lesion = {lesion_of(i) for i in tr_img}
    te_lesion = {lesion_of(i) for i in te_img}
    lesion_overlap = tr_lesion & te_lesion

    lines = []
    lines.append(f"数据总数: {len(data)}  可用样本: {len(pool)}  跳过: {dict(skipped)}")
    lines.append(f"划分比例(训练图片占比目标): {args.ratio}  seed: {args.seed}")
    lines.append(f"训练集: {len(train_items)} 条  评估集: {len(test_items)} 条")
    lines.append("")
    lines.append(f"{'dx':<8}{'病灶数':<8}{'训练(图)':<10}{'评估(图)':<10}")
    for dx in sorted(by_dx):
        lines.append(f"{dx:<8}{len(by_dx[dx]):<8}{tr_cnt.get(dx, 0):<10}{te_cnt.get(dx, 0):<10}")
    lines.append("")
    lines.append("重叠校验（互斥性证明）：")
    lines.append(f"  图片级重叠: {len(img_overlap)}（应为 0）")
    lines.append(f"  病灶级(lesion_id)重叠: {len(lesion_overlap)}（应为 0）")
    lines.append("  说明：同一病灶的多张影像被强制划入同一侧，杜绝近重复泄漏。")
    report = "\n".join(lines)
    (out_dir / "split_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n已写出: {out_dir / 'skin_vqa_train_split.json'}")
    print(f"        {out_dir / 'skin_vqa_test_split.json'}")
    print(f"        {out_dir / 'split_report.txt'}")


if __name__ == "__main__":
    main()