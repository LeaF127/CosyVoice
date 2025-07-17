#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=1
stop_stage=1

data_dir=/root/gpufree-data/data
pretrained_model_dir=../../pretrained_models/CosyVoice2-0.5B
# 用来复制
TRAIN_DATASETS=(
  "/root/gpufree-data/data/LibriTTS/train-clean-100/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/LibriTTS/test-clean/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/LibriTTS/test-other/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/LibriTTS/dev-clean/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/biaobei/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/HWB_modified/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/jiangpeng/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/minnan-female/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/minnan-male/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/taiwan-hokkien/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/vivos/train/cosyvoice_map_folder/parquet/data.list"
)
DEV_DATASETS=(
  "/root/gpufree-data/data/LibriTTS/dev-other/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/minnan/zhongxun/cosyvoice_map_folder/parquet/data.list"
  "/root/gpufree-data/data/vivos/test/cosyvoice_map_folder/parquet/data.list"
)


# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=1
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # 将几个数据集的数据文件连接起来
  for file in "${TRAIN_DATASETS[@]}"; do
    cat "$file" >> data/train.data.list
  done
  for file in "${DEV_DATASETS[@]}"; do
    cat "$file" >> data/dev.data.list
  done
  # NOTE will update llm/hift training later
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi