#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=/root/gpufree-data/data
pretrained_model_dir=../../pretrained_models/CosyVoice2-0.5B

# 下载libritts语料库
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  mkdir -p $data_dir
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

# 生成数据映射文件
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in dev-clean test-clean dev-other test-other train-clean-100; do
    mkdir -p $data_dir/LibriTTS/$x/cosyvoice_map_folder
    python local/prepare_libritts_data.py --src_dir $data_dir/LibriTTS/$x --des_dir $data_dir/LibriTTS/$x/cosyvoice_map_folder
  done
  for x in biaobei HWB_modified jiangpeng minnan-female minnan-male taiwan-hokkien zhongxun; do
    mkdir -p $data_dir/minnan/$x/cosyvoice_map_folder
    python local/prepare_data.py --src_dir $data_dir/minnan/$x --des_dir $data_dir/minnan/$x/cosyvoice_map_folder
  done
  for x in test train; do
    mkdir -p $data_dir/vivos/$x/cosyvoice_map_folder
    python local/prepare_data.py --src_dir $data_dir/vivos/$x --des_dir $data_dir/vivos/$x/cosyvoice_map_folder
  done 
fi

# 提取音频的embedding，并与utt和spk做映射
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in cosyvoice_map_folder"
  for x in dev-clean test-clean dev-other test-other train-clean-100; do
    tools/extract_embedding.py --dir $data_dir/LibriTTS/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
  for x in biaobei HWB_modified jiangpeng minnan-female minnan-male taiwan-hokkien zhongxun; do
    tools/extract_embedding.py --dir $data_dir/minnan/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
  for x in test train; do
    tools/extract_embedding.py --dir $data_dir/vivos/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# 提取离散语音token
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in cosyvoice_map_folder"
  for x in dev-clean test-clean dev-other test-other train-clean-100; do
    tools/extract_speech_token.py --dir $data_dir/LibriTTS/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
  for x in biaobei HWB_modified jiangpeng minnan-female minnan-male taiwan-hokkien zhongxun; do
    tools/extract_speech_token.py --dir $data_dir/minnan/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
  for x in test train; do
    tools/extract_speech_token.py --dir $data_dir/vivos/$x/cosyvoice_map_folder \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

# 生成Parquet文件准备训练
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in dev-clean test-clean dev-other test-other train-clean-100; do
    mkdir -p $data_dir/LibriTTS/$x/cosyvoice_map_folder/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/LibriTTS/$x/cosyvoice_map_folder \
      --des_dir $data_dir/LibriTTS/$x/cosyvoice_map_folder/parquet
  done
  for x in biaobei HWB_modified jiangpeng minnan-female minnan-male taiwan-hokkien zhongxun; do
    mkdir -p $data_dir/minnan/$x/cosyvoice_map_folder/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/minnan/$x/cosyvoice_map_folder \
      --des_dir $data_dir/minnan/$x/cosyvoice_map_folder/parquet
  done
  for x in test train; do
    mkdir -p $data_dir/vivos/$x/cosyvoice_map_folder/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/vivos/$x/cosyvoice_map_folder \
      --des_dir $data_dir/vivos/$x/cosyvoice_map_folder/parquet
  done
fi