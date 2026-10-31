# -*- coding: utf-8 -*-
"""把 LoRA 适配器合并进基座模型，导出独立权重目录，供 vLLM 直接加载。
在 CPU 上合并，不占 GPU 显存。

用法：
    python merge_lora.py \
        --base    /path/to/Qwen2.5-VL-7B-Instruct \
        --adapter /path/to/adapters/rank16_balanced \
        --out     /path/to/models/merged-r16-bal
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基座模型")
    parser.add_argument("--base", required=True, help="基座模型目录")
    parser.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    parser.add_argument("--out", required=True, help="合并后权重输出目录")
    args = parser.parse_args()

    BASE = args.base
    ADAPTER = args.adapter
    OUT = args.out

    print(f"基座  : {BASE}", flush=True)
    print(f"适配器: {ADAPTER}", flush=True)
    print(f"输出  : {OUT}", flush=True)

    print("加载基座(CPU)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu")

    print("挂载 LoRA...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)

    print("合并权重...", flush=True)
    model = model.merge_and_unload()

    print("保存合并模型...", flush=True)
    model.save_pretrained(OUT, safe_serialization=True)

    AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
    print("DONE:", OUT, flush=True)


if __name__ == "__main__":
    main()