# -*- coding: utf-8 -*-
"""皮肤 VQA 类别平衡重训脚本（P0：A1 Focal Loss + B1 类别反比采样）。

针对 rank16 全量评估暴露的少数类问题（df 0%、mel 36.6%、akiec 28.6%），
在原 LoRA SFT 基础上叠加两项训练侧改动：

  - B1 平衡采样：WeightedRandomSampler 按「类别频率反比」采样，
    让每个 batch 内 df/mel/akiec 等少数类被充分看到（承担类别平衡主责）。
  - A1 Focal Loss：在因果语言建模交叉熵上套 Focal（γ 可调），
    聚焦难分样本，进一步放大少数类的梯度贡献（per-token，起辅助作用）。

训练数据沿用 LLaMA-Factory 的 skin_vqa_train.json（conversations 格式，
类别标签藏在 assistant 回答的「诊断为：XXX」里，脚本内自动解析为 dx）。

用法（在 AutoDL 服务器上，先停掉 vLLM 释放显存）：
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python train_balanced.py \
        --model_name /path/to/Qwen2.5-VL-7B-Instruct \
        --train_data /path/to/skin_vqa_train_split.json \
        --image_dir  /path/to/images \
        --output_dir /path/to/adapters/rank16_balanced

训练完成后：
    1) 用 merge_lora.py 把 adapter 合并到基座模型；
    2) 停旧 vLLM、用合并后权重重启 vLLM；
    3) 用 eval.py --adapter <output_dir> 复测，对照 df/mel/akiec 召回。
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# 病名 → 代码（与 data_converter.py / eval.py 保持一致）
DX_CN2CODE = {
    "黑色素细胞痣": "nv",
    "黑色素瘤": "mel",
    "良性角化病": "bkl",
    "基底细胞癌": "bcc",
    "光化性角化病": "akiec",
    "血管病变": "vasc",
    "皮肤纤维瘤": "df",
}


def extract_dx(answer: str):
    """从「诊断为：XXX。」的 answer 中解析诊断代码；解析不出返回 None。"""
    m = re.search(r"诊断为[：:]\s*(.+?)[。\n]", answer)
    seg = m.group(1).strip() if m else answer
    for cn, code in DX_CN2CODE.items():
        if cn in seg:
            return code
    return None


class SkinVQADataset(Dataset):
    """把 LLaMA-Factory 的 conversations 转成 Qwen2.5-VL 消息格式。

    加载时一次性解析每条样本的 dx（存 self.labels，供采样器用），
    __getitem__ 只返回构造好的 messages，tokenize 交给 collator 批量处理。
    """

    def __init__(self, data, processor, image_dir: str):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.records = []
        self.labels = []
        for item in data:
            dx, messages = self._build(item)
            if dx is None:
                continue
            self.records.append(messages)
            self.labels.append(dx)

    def _build(self, item):
        imgs = item.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        if len(imgs) != 1:
            return None, None
        img_path = str(self.image_dir / Path(imgs[0]).name)

        conv = item.get("conversations") or []
        if len(conv) < 2:
            return None, None
        question = conv[0].get("value", "").replace("<image>", "").strip()
        answer = conv[1].get("value", "")
        dx = extract_dx(answer)
        if dx is None:
            return None, None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        return dx, messages

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {"messages": self.records[idx]}


class DataCollator:
    """批量 tokenize 并构造 labels（只对 assistant 部分计算 loss）。"""

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        full_msgs = [f["messages"] for f in features]
        prompt_msgs = [m[:1] for m in full_msgs]  # 只保留 user 轮

        full_texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in full_msgs
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in prompt_msgs
        ]

        # 图片只出现在 user 轮，prompt 与 full 共用同一份 image_inputs 即可
        image_inputs, video_inputs = process_vision_info(full_msgs)

        batch = self.processor(
            text=full_texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        prompt_ids = self.processor(
            text=prompt_texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )["input_ids"]

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        # 每条样本把 prompt 部分（含视觉 token）mask 掉，只对回答算 loss
        for i in range(len(features)):
            plen = int((prompt_ids[i] != self.pad_token_id).sum())
            labels[i, :plen] = -100
        labels[input_ids == self.pad_token_id] = -100
        batch["labels"] = labels
        return batch


class BalancedSamplerMixin:
    """支持注入 WeightedRandomSampler（类别平衡采样）。

    transformers.Trainer 不接受 train_sampler 形参，需在 get_train_dataloader
    里把默认 RandomSampler 替换为平衡采样器（单卡场景标准做法）。
    """

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_sampler = train_sampler

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        if self.custom_train_sampler is not None:
            dl.sampler = self.custom_train_sampler
        return dl


class BalancedTrainer(BalancedSamplerMixin, Trainer):
    """普通交叉熵 + 平衡采样（Focal Loss 消融对照用）。"""


class FocalTrainer(BalancedSamplerMixin, Trainer):
    """在因果语言建模交叉熵上套 Focal Loss，聚焦难分样本。

    gamma: 聚焦因子，2.0 为常用值；0 等价于普通交叉熵（可做消融对照）。
    """

    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.shape[1] > labels.shape[1]:
            logits = logits[:, : labels.shape[1], :]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_labels = shift_labels.view(-1)
        mask = flat_labels != -100

        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            flat_labels, reduction="none",
        )  # -100 位置自动输出 0
        if self.gamma == 0:
            loss = ce.sum() / mask.sum().clamp(min=1)
        else:
            pt = torch.exp(-ce).clamp(min=1e-8)
            focal = torch.where(mask, (1 - pt) ** self.gamma * ce, ce)
            loss = focal.sum() / mask.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def find_all_linear_names(model):
    """返回语言模型所有线性层类型名（排除视觉塔 visual）。"""
    lm_module_names = set()
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            lm_module_names.add(name.split(".")[-1])
    return list(lm_module_names)


def build_balanced_sampler(labels):
    """按类别频率反比构造采样器：w_i = N / (K * n_i)。"""
    cnt = Counter(labels)
    n, k = len(labels), len(cnt)
    class_w = {c: n / (k * v) for c, v in cnt.items()}
    per_sample_w = [class_w[lab] for lab in labels]
    print("类别样本数:", dict(sorted(cnt.items())))
    print("类别采样权重:", {c: round(w, 3) for c, w in class_w.items()})
    return WeightedRandomSampler(per_sample_w, num_samples=n, replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="基座模型目录")
    ap.add_argument("--train_data", required=True, help="skin_vqa_train.json 路径")
    ap.add_argument("--image_dir", default=None, help="图片目录，默认取 train_data 同目录 images/")
    ap.add_argument("--output_dir", default="adapters/rank16_balanced")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--image_min_pixels", type=int, default=1024)
    ap.add_argument("--image_max_pixels", type=int, default=768 * 768)
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal 聚焦因子，0 表示普通交叉熵（消融用）")
    ap.add_argument("--no_balance", action="store_true",
                    help="关闭类别平衡采样（消融对照用）")
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    help="注意力实现；无 flash-attn 时改为 eager")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，固定以保证训练可复现（与 split_train_test 共用 42）")
    args = ap.parse_args()

    if args.image_dir is None:
        args.image_dir = str(Path(args.train_data).parent / "images")

    # 固定随机种子：DataLoader shuffle / 采样 / 初始化均出可复现结果
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1) 数据与类别统计
    raw = json.loads(Path(args.train_data).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dataset = SkinVQADataset(raw, processor, args.image_dir)
    print(f"可用训练样本: {len(dataset)} / 原始 {len(raw)}")
    if len(dataset) == 0:
        raise SystemExit("没有可用样本，请检查 train_data 与 image_dir 路径")

    train_sampler = None
    if not args.no_balance:
        train_sampler = build_balanced_sampler(dataset.labels)

    # 2) 模型 + 冻结视觉塔/投影层 + LoRA
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    # 冻结视觉编码器 + 多模态投影层（与原 LLaMA-Factory 配置一致）
    for p in model.visual.parameters():
        p.requires_grad = False

    target_modules = find_all_linear_names(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # 冻结视觉塔 + 梯度检查点必需
    model.config.use_cache = False
    model.print_trainable_parameters()

    # 3) 训练
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        save_total_limit=2,
    )

    trainer_cls = BalancedTrainer if args.focal_gamma == 0 else FocalTrainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(processor),
    )
    if trainer_cls is FocalTrainer:
        trainer_kwargs["gamma"] = args.focal_gamma
    if train_sampler is not None:
        trainer_kwargs["train_sampler"] = train_sampler

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    # 4) 保存 adapter
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n完成！adapter 已保存到: {args.output_dir}")
    print("后续步骤：")
    print("  1) 用 merge_lora.py 把 adapter 合并到基座模型；")
    print("  2) 停旧 vLLM，用合并后权重重启 vLLM；")
    print("  3) 用 eval.py --adapter <output_dir> 复测 df/mel/akiec 召回。")


if __name__ == "__main__":
    main()