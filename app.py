import os, time, sys, shutil, io
from pathlib import Path
from flask import Flask, request, jsonify, send_file, make_response, Response
from flask_cors import CORS
import requests
import subprocess
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from typing import Union

# vllm
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# 固定参数
root_dir = Path(__file__).parent.as_posix()
host = "0.0.0.0"
port = 9396

# ffmpeg 环境变量（确保运行时可以找到）
if sys.platform == "win32":
    os.environ['PATH'] = root_dir + f';{root_dir}\\ffmpeg;' + os.environ['PATH']+f';{root_dir}/third_party/Matcha-TTS'
else:
    os.environ['PATH'] = root_dir + f':{root_dir}/ffmpeg:' + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS') # 添加第三方组件 Matcha-TTS

# Flask 配置
app = Flask(__name__)
# CORS(app)

# 解析请求参数
def get_params(req):
    text = req.args.get("text", "").strip() or req.form.get("text", "").strip()
    speed = float(req.form.get("speed", 1.0))
    
    # 支持上传音频/url
    if 'wav_file' in req.files:
        file = req.files['wav_file']
        reference_audio = io.BytesIO(file.read())
    # 从url获取prompt音频
    elif 'wav_url' in req.form:
        url = req.form.get("wav_url", "")
        if url.startwith("https://") or url.startwith("http://"):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                reference_audio = io.BytesIO(r.content)
                print(f"[INFO] 成功下载提示音频：{url}")
            except Exception as e:
                raise RuntimeError(f"下载参考音频失败：{e}")
    else:
        reference_audio = req.args.get("reference_audio") or req.form.get("reference_audio")  
    
    if not reference_audio:
        reference_audio = os.path.join(root_dir, "asset/zero_shot_prompt.wav")
        print("未传入参考音频，使用默认参考音色")

    return {
        "text": text,
        "reference_audio": reference_audio,
        "speed": speed
    }

def convert_audio_to_16k(reference_audio: Union[str, io.BytesIO]) -> io.BytesIO:
    """
    使用 ffmpeg 将任意格式的参考音频转换为 16kHz 单声道，并写入 BytesIO 对象。
    防止保存为本地文件。
    """
    # 从二进制流中读取音频
    if isinstance(reference_audio, io.BytesIO):
        reference_audio.seek(0)
        input_data = reference_audio.read()
        input_pipe = "pipe:0"
    else:
        input_data = None
        input_pipe = reference_audio # 默认音频路径
    
    try:
        process = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-i", input_pipe,
                "-ar", "16000",
                "-ac", "1",   # 强制单声道
                "-f", "wav",  # 强制输出格式为 wav
                "pipe:1"      # 输出到 stdout
            ],
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        audio_bytes = process.stdout
        return io.BytesIO(audio_bytes)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 转换失败: {e.stderr.decode()}")
        raise RuntimeError("ffmpeg 转换参考音频失败")


# 合成过程
def batch(params):
    global tts_model
    if not shutil.which("ffmpeg"):
        raise RuntimeError("系统未安装 ffmpeg")
    
    if isinstance(params["reference_audio"], str):
        if not os.path.exists(params["reference_audio"]):
            raise FileNotFoundError(f"参考音频不存在: {params['reference_audio']}")
        
    # 加载音频并调用模型合成    
    try:
        buffer = convert_audio_to_16k(params["reference_audio"])
        prompt = load_wav(buffer, target_sr=16000)
        # prompt = load_wav(params['reference_audio'], target_sr=16000)
        instruct = "请用自然的语气说这句话"
        audio_list = []
        for out in tts_model.inference_instruct2(params["text"], instruct, prompt, stream=False, speed=params["speed"]):
            audio_list.append(out["tts_speech"])
        audio = torch.cat(audio_list, dim=1)
        
        import io
        buffer = io.BytesIO() # 使用内存，避免在服务器生成临时文件
        torchaudio.save(buffer, audio, 24000, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"[ERROR] 合成音频失败: {e}")
    
# 主接口
@app.route("/cv2tts", methods=["POST", "GET"])
def tts():
    try:
        # 处理request，获取params
        params = get_params(request)
        if not params["text"]:
            return make_response(jsonify({"code": 6, "msg": "缺少文本"}), 500)
        buffer = batch(params)

        return send_file(
            buffer,
            mimetype="audio/x-wav",
            as_attachment=False,
            download_name="tts_output.wav"
        )

    except Exception as e:
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)

# 启动服务
if __name__ == "__main__":
    tts_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    print(f"🚀 接口启动成功：http://{host}:{port}/tts")
    
    app.run(host=host, port=port)

# TODO 
# - 改为流式输出
# - 训中英闽模型然后替换
# - vllm加速