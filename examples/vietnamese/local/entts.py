import code
import datetime
from genericpath import exists
import json
import os
import sys

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage

with open('三虾面utt2text.json', 'r', encoding='utf-8') as f:
    txt = json.load(f)
tts_text = txt['english']

for utt, tts_text in tts_text.items():
    prompt_speech_16k = load_wav(f'asset/3xm/{utt}.wav', 16000, duration = 15)   
    audio_list = []
    for _, model_output in enumerate(cosyvoice.inference_instruct2(tts_text=tts_text,
                                                        instruct_text='用自然的语气说这句话',
                                                        prompt_speech_16k=prompt_speech_16k,
                                                        text_frontend=True)):
        audio_list.append(model_output['tts_speech'])        
    audio = torch.cat(audio_list, dim=1)
    os.makedirs('tmp/3xm/en', exist_ok=True)
    torchaudio.save(f'tmp/3xm/en/{utt}.wav', audio, cosyvoice.sample_rate)
