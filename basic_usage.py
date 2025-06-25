import datetime
import sys

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('asset\cross_lingual_prompt.wav', 16000)

for i, j in enumerate(cosyvoice.inference_instruct2(tts_text='我国东北的小兴安岭，有数不清的树，几百里连成一片，就像绿色的海洋。',
                                                    instruct_text='用伤心的语气说这句话',
                                                    prompt_speech_16k=prompt_speech_16k)):
    torchaudio.save(f'tmp/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S-bs")}.wav', j['tts_speech'], cosyvoice.sample_rate)
