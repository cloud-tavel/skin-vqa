#!/usr/bin/env bash
# =============================================================================
# 皮肤 VQA · A1+B1 类别平衡重训 一键脚本
# 用法（在仓库根目录下；BASE 默认 /root/autodl-tmp，可用环境变量覆盖）：
#   bash run_retrain.sh stop    # 1) 停 vLLM + 后端，释放显存
#   bash run_retrain.sh train   # 2) 类别平衡重训（后台 nohup，日志 logs/train.log）
#   bash run_retrain.sh merge   # 3) 合并 LoRA 到基座（生成 merged-r16-bal）
#   bash run_retrain.sh serve   # 4) 重启 vLLM + FastAPI 后端
#   bash run_retrain.sh eval    # 5) 全量复测评估（后台，日志 logs/eval.log）
#   bash run_retrain.sh all     # 顺序执行 1→2→3（train 完成后需手动确认再 serve/eval）
#   bash run_retrain.sh status  # 查看训练/评估日志尾部 + GPU 状态
#
# ⚠️ 训练约需数小时（4090 / 10015 样本 / 3 epoch）。train 阶段提交后台后即可断开 SSH。
# =============================================================================
set -uo pipefail

# ------------------------- 路径配置（可用环境变量覆盖） -------------------------
BASE=${BASE:-/root/autodl-tmp}
MODEL_DIR=${MODEL_DIR:-$BASE/models/Qwen2.5-VL-7B-Instruct}     # 基座模型目录
TRAIN_DATA=${TRAIN_DATA:-$BASE/data/skin_vqa_train_split.json}  # 病灶级互斥训练集
IMAGE_DIR=${IMAGE_DIR:-$BASE/images}                            # 训练图片目录
ADAPTER_OUT=${ADAPTER_OUT:-$BASE/models/adapters/rank16_balanced}
MERGED_OUT=${MERGED_OUT:-$BASE/models/merged-r16-bal}
MERGE_SCRIPT=$BASE/deploy/merge_lora.py
EVAL_SCRIPT=$BASE/eval/eval.py
VLLM_PORT=${VLLM_PORT:-8000}
API_PORT=${API_PORT:-6006}
EXTRA_TRAIN_ARGS=""                                    # 例如: --focal_gamma 2.0

cd "$BASE" || exit 1
mkdir -p logs

stop_services() {
  echo "== 停止 vLLM / 后端 / 评估进程 =="
  pkill -f "vllm" 2>/dev/null && echo "vllm 已停止" || echo "无 vllm 进程"
  pkill -f "uvicorn" 2>/dev/null && echo "uvicorn 已停止" || echo "无 uvicorn 进程"
  pkill -f "eval.py" 2>/dev/null || true
  sleep 8
  echo "== 当前 GPU 状态 =="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
}

do_train() {
  # 数据存在性检查：训练数据与图片目录缺一即中止，避免空跑
  if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误：训练数据 $TRAIN_DATA 不存在，请先上传 skin_vqa_train.json"; exit 1
  fi
  if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误：图片目录 $IMAGE_DIR 不存在，请先上传 images/"; exit 1
  fi
  stop_services
  echo "== 启动类别平衡重训 (A1 Focal + B1 平衡采样) =="
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup python "$BASE/train/train_balanced.py" \
    --model_name "$MODEL_DIR" \
    --train_data "$TRAIN_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$ADAPTER_OUT" \
    --epochs 3 --batch_size 2 --grad_accum 8 --lr 5e-5 \
    --lora_r 16 --lora_alpha 32 --focal_gamma 2.0 \
    $EXTRA_TRAIN_ARGS \
    > logs/train_balanced.log 2>&1 &
  echo "训练已后台启动，PID=$!"
  echo "查看进度: tail -f $BASE/logs/train_balanced.log"
}

do_merge() {
  echo "== 合并 LoRA 权重 =="
  python "$MERGE_SCRIPT" \
    --base "$MODEL_DIR" --adapter "$ADAPTER_OUT" --out "$MERGED_OUT"
  echo "合并完成: $MERGED_OUT"
  ls -lh "$MERGED_OUT"
}

do_serve() {
  stop_services
  echo "== 启动 vLLM =="
  nohup vllm serve "$MERGED_OUT" \
    --port $VLLM_PORT --max-model-len 4096 \
    --served-model-name skin-vqa \
    --gpu-memory-utilization 0.90 \
    > logs/vllm.log 2>&1 &
  echo "vLLM 启动中（约 1-2 分钟），日志: logs/vllm.log"
  sleep 90
  echo "== 启动 FastAPI 后端 =="
  cd "$BASE/deploy" || exit 1
  nohup env TENCENT_SECRET_ID="${TENCENT_SECRET_ID:-}" \
            TENCENT_SECRET_KEY="${TENCENT_SECRET_KEY:-}" \
    uvicorn main:app --host 0.0.0.0 --port $API_PORT \
    > "$BASE/logs/server.log" 2>&1 &
  cd "$BASE" || exit 1
  echo "后端已启动（端口 $API_PORT）。若配置了腾讯云密钥，请先 export 后重跑本命令。"
  echo "== 健康检查 =="
  sleep 5
  curl -s http://127.0.0.1:$API_PORT/health || echo "（稍等片刻后手动 curl 重试）"
}

do_eval() {
  stop_services
  echo "== 互斥测试集复评（新 adapter，1573 例零重叠） =="
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup python "$EVAL_SCRIPT" \
    --test "$BASE/data/skin_vqa_test_split.json" \
    --adapter "$ADAPTER_OUT" \
    --out "$BASE/split_results.csv" \
    > logs/eval_balanced.log 2>&1 &
  echo "评估已后台启动，PID=$!"
  echo "查看进度: tail -f $BASE/logs/eval_balanced.log"
}

show_status() {
  echo "== 进程 =="
  ps aux | grep -E "train_balanced|eval.py|vllm|uvicorn" | grep -v grep || echo "（无相关进程）"
  echo "== GPU =="
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  echo "== 训练日志尾部 =="
  tail -n 15 logs/train_balanced.log 2>/dev/null || echo "（无训练日志）"
  echo "== 评估日志尾部 =="
  tail -n 15 logs/eval_balanced.log 2>/dev/null || echo "（无评估日志）"
}

case "${1:-help}" in
  stop)   stop_services ;;
  train)  do_train ;;
  merge)  do_merge ;;
  serve)  do_serve ;;
  eval)   do_eval ;;
  status) show_status ;;
  all)    do_train; echo "训练已后台运行；完成后执行: bash run_retrain.sh merge && bash run_retrain.sh serve && bash run_retrain.sh eval" ;;
  *)      grep '^#' "$0" | head -n 16 ;;
esac
