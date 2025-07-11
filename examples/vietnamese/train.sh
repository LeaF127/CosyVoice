#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=0

data_dir=./data/vivos
pretrained_model_dir=../../pretrained_models/CosyVoice2-0.5B

# inference
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  # TODO consider remove bin/inference.py, or use similar initilization method as in readme
  for mode in sft; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice2.yaml \
      --prompt_data $data_dir/test/cosyvoice_map_folder/parquet/data.list \
      --prompt_utt2data $data_dir/test/cosyvoice_map_folder/parquet/utt2data.list \
      --tts_text `pwd`/tts_text.json \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --llm_model $pretrained_model_dir/llm_vie.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice2//$mode
  done
fi

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
  # cat $data_dir/train/cosyvoice_map_folder/parquet/data.list > data/train.data.list
  # cat $data_dir/test/cosyvoice_map_folder/parquet/data.list > data/dev.data.list
  # NOTE will update llm/hift training later
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data $data_dir/train/cosyvoice_map_folder/parquet/data.list \
      --cv_data $data_dir/test/cosyvoice_map_folder/parquet/data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
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
average_num=9
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi