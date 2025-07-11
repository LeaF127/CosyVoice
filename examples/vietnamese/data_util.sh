#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=1
stop_stage=3

data_dir=./data/vivos
pretrained_model_dir=../../pretrained_models/CosyVoice2-0.5B

# 生成数据映射文件
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in test train; do
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir $data_dir/$x/cosyvoice_map_folder
  done
fi

# 提取音频的embedding，并与utt和spk做映射
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in test train; do
    echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in $data_dir/$x/cosyvoice_map_folder dir"
    tools/extract_embedding.py --dir $data_dir/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# 提取离散语音token
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
  for x in test train; do
    echo "Extract discrete speech token, you will get utt2speech_token.pt in $data_dir/$x/cosyvoice_map_folder dir"
    tools/extract_speech_token.py --dir $data_dir/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

# 生成Parquet文件准备训练
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for x in test train; do
    echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt in $data_dir/$x/cosyvoice_map_folder dir"
    mkdir -p $data_dir/$x/cosyvoice_map_folder/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/$x/cosyvoice_map_folder \
      --des_dir $data_dir/$x/cosyvoice_map_folder/parquet
  done
fi