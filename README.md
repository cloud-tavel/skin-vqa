# Skin-VQA：皮肤病变视觉语言问答系统

基于 **Qwen2.5-VL-7B** 的多模态皮肤镜问答系统：输入一张皮肤镜图像和一句自然语言问题，输出结构化诊断与建议。覆盖 LoRA 微调、类别不平衡重训、vLLM 推理加速、FastAPI 服务与微信小程序前端。

> **大学生创新训练计划项目**（校级一般项目 · 江南大学 · 人工智能专业）
> 项目负责人：张景皓　指导教师：潘祥 副教授

## 效果

### 诊断准确率（7 类皮肤病变）

| 配置 | 测试集 | micro | macro | 说明 |
|---|---|---|---|---|
| Qwen2.5-VL-7B + LoRA rank16 | 505 例（与训练集同源） | 71.88% | 48.59% | 早期基线，指标偏乐观 |
| ＋ 类别平衡重训（Focal Loss + 加权采样） | 1573 例（病灶级互斥） | **74.25%** | **55.81%** | 训练 6299 / 测试 1573，图片级与病灶级重叠均为 0 |

互斥测试集上的类别召回变化：**df 0% → 39.1%**，akiec +20.6pp，mel +7.8pp，nv 92.8% → 93.1%。

### 推理延迟

| 推理后端 | 平均 | P50 |
|---|---|---|
| transformers（逐条生成） | 3.20 s | — |
| vLLM 0.8.5 + 合并 LoRA 权重 | **1.81 s** | **1.57 s** |

20 样本在线压测，稳态区间 1.4–1.9 s（首条 5.18 s 为冷启动）。

## 系统架构

```
┌────────────────────────┐
│ 微信小程序（原生）      │  拍照 / 相册 / 语音输入 / 多轮历史
└───────────┬────────────┘
            │ multipart: image + question + history
            ▼
┌────────────────────────┐
│ FastAPI  :6006         │  /ask   /asr   /health
└───────────┬────────────┘
            │ OpenAI 兼容协议
            ▼
┌────────────────────────┐
│ vLLM     :8000         │  merged Qwen2.5-VL-7B + LoRA
└────────────────────────┘
```

`deploy/direct_infer.py` 是不依赖 vLLM 的直推版本（transformers + PEFT 运行时挂载适配器），适合单卡调试与功能验证。

## 仓库结构

```
skin-vqa/
├── data/
│   ├── build_dataset.py        # HAM10000 标注 → conversations 训练集
│   └── split_train_test.py     # 按 lesion_id 做病灶级互斥 8:2 划分
├── train/
│   ├── train_balanced.py       # LoRA 微调 + Focal Loss + 加权采样
│   └── run_retrain.sh          # 一键：stop → train → merge → serve → eval
├── eval/
│   ├── eval.py                 # 诊断准确率评估（micro / macro / 分类别）
│   └── bench_latency.py        # vLLM 在线延迟基准
├── deploy/
│   ├── main.py                 # FastAPI：转发 vLLM OpenAI 兼容接口
│   ├── direct_infer.py         # FastAPI：transformers 直推（无 vLLM）
│   ├── merge_lora.py           # LoRA 合并进基座（CPU 上执行）
│   └── requirements-server.txt
├── miniprogram/                # 微信小程序前端
└── docs/
    └── experiments.md          # 实验记录与结论
```

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
# 需要 vLLM 部署时另行安装（与 CUDA 版本匹配）：
# pip install vllm==0.8.5.post1
```

推荐 Python 3.10、PyTorch 2.4+、`transformers==4.51.3`。

### 2. 数据准备

本仓库**不附带任何图像数据**。请自行获取 [HAM10000](https://www.isic-archive.com/) 数据集（遵循其原始许可，通常为 CC BY-NC），再由 `data/build_dataset.py` 生成训练语料：

```bash
python data/build_dataset.py \
    --meta HAM10000_metadata.csv \
    --image_dir images \
    --out skin_vqa_train.json --seed 42
```

脚本为 7 个类别套用统一的「特征描述 + 医疗建议」模板，并从 6 条问法中按种子随机抽一条，产出结构如下（单图、两轮对话）：

```json
[
  {
    "id": "0",
    "image": ["images/ISIC_0024306.jpg"],
    "conversations": [
      {"from": "user",      "value": "<image>这是什么皮肤病？需要去医院吗？"},
      {"from": "assistant", "value": "根据图像特征分析，该皮损倾向于诊断为：黑色素细胞痣 (Melanocytic nevi)。\n\n【特征描述】：……\n【医疗建议】：……"}
    ]
  }
]
```

约定：单图；`conversations` 恰好两轮；assistant 回答中包含「诊断为：XXX」，下游脚本据此解析类别标签（改动措辞需同步核对 `train/train_balanced.py` 与 `data/split_train_test.py` 的正则）。

完成病灶级互斥划分：

```bash
python data/split_train_test.py \
    --data skin_vqa_train.json \
    --meta HAM10000_metadata.csv \
    --image_dir images \
    --out split --ratio 0.8 --seed 42
```

输出 `skin_vqa_train_split.json`、`skin_vqa_test_split.json` 与划分报告；同一 `lesion_id` 的所有影像只会进入同一侧，避免数据泄漏。

### 3. 训练

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train/train_balanced.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --train_data split/skin_vqa_train_split.json \
    --image_dir images \
    --output_dir adapters/rank16_balanced \
    --lora_r 16 --lora_alpha 32 \
    --focal_gamma 2.0 \
    --epochs 3 --batch_size 2 --grad_accum 8 --lr 5e-5
```

消融开关：`--no_balance` 关闭加权采样，`--focal_gamma 0` 关闭 Focal Loss。

### 4. 合并与部署

```bash
python deploy/merge_lora.py \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter adapters/rank16_balanced \
    --out models/merged-r16-bal

vllm serve models/merged-r16-bal \
    --port 8000 --max-model-len 4096 \
    --served-model-name skin-vqa \
    --gpu-memory-utilization 0.90

cd deploy && uvicorn main:app --host 0.0.0.0 --port 6006
```

服务地址支持环境变量覆盖：`VLLM_CHAT`、`VLLM_MODELS`、`VLLM_MODEL_NAME`。

### 5. 评估与延迟基准

```bash
python eval/eval.py \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter adapters/rank16_balanced \
    --test split/skin_vqa_test_split.json \
    --out results.csv

python eval/bench_latency.py --n 20
```

### 6. 小程序

用微信开发者工具打开 `miniprogram/` 目录，`appid` 已设为游客模式。修改 `miniprogram/app.js` 中的 `baseUrl` 指向你的后端地址；真机调试需在小程序后台配置合法域名。

## 关键实现

- **LoRA 微调**：r=16、α=32，注入语言塔线性层与视觉塔 MLP；单卡 RTX 4090（24GB）3 epoch 约 93 分钟。
- **类别不平衡**：`WeightedRandomSampler` 按类别频率反比采样为主，Focal Loss（γ=2.0）为辅，针对 df/mel/akiec 长尾类。
- **推理加速**：LoRA 权重合并进基座后用 vLLM 常驻服务，延迟从 3.20 s 降至 1.81 s，满足 ≤2 s 的实时问答要求。
- **诊断解析**：优先匹配「诊断为：」后的精确病名，未命中时按强特异性关键词兜底，评估中无法解析率为 0。
- **数据泄漏修复**：早期测试集由训练集分层抽样而来，指标偏乐观；现按 `lesion_id` 做病灶级互斥划分后重新评估。
- **负结果记录**：推理侧注入黑色素瘤鉴别提示词会使 micro 从 71.88% 跌至 31.68%，少数类问题必须回到训练侧解决（见实验记录 E5）。

## 实验记录

完整的实验设计、逐类指标、消融结果与工程踩坑见 [docs/experiments.md](docs/experiments.md)。

## 局限

- 训练语料为模板化构造，真实临床场景的泛化能力尚未验证；
- df、mel 等长尾类可用样本偏少（df 仅 92 张），macro 指标仍有提升空间；
- 仅覆盖 7 类皮肤病变，不构成鉴别诊断的完整集合；
- **本项目为教学与科研用途，不能用于临床诊断，也不构成医疗器械**。

## 许可与声明

- **代码**：MIT，见 [LICENSE](LICENSE)。
- **数据**：HAM10000 / ISIC 图像与标注不随本仓库分发，使用时须遵守其原始许可（通常为 CC BY-NC，禁止商业用途）。
- **模型**：Qwen2.5-VL-7B-Instruct 遵循其上游许可；LoRA 权重与合并权重未随仓库提供。
- 本项目仅用于学习与研究，不得用于商业用途或临床决策。

## 致谢

感谢指导教师潘祥副教授在选题与技术路线上的指导，感谢学校提供的训练经费与算力支持，以及 Qwen 团队与 HAM10000 数据集的开放工作。
