import os,time,sys
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
root_dir=Path(__file__).parent.as_posix()

# ffmpeg
if sys.platform == 'win32':
    os.environ['PATH'] = root_dir + f';{root_dir}\\ffmpeg;' + os.environ['PATH']+f';{root_dir}/third_party/Matcha-TTS'
else:
    os.environ['PATH'] = root_dir + f':{root_dir}/ffmpeg:' + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')
tmp_dir=Path(f'{root_dir}/tmp').as_posix()
logs_dir=Path(f'{root_dir}/logs').as_posix()
os.makedirs(tmp_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)

from flask import Flask, request, jsonify, send_file, Response, make_response,send_file
import logging
import subprocess
import shutil
import datetime
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# FunASR
from funasr import AutoModel
asr_model = AutoModel(
    model="D:\seki\work\CosyVoice\pretrained_models\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    
    device="cuda:0",
    disable_update=True
)

# ArgosTranslate
import argostranslate.package
import argostranslate.translate


import torchaudio,torch
from pathlib import Path
import base64

'''
app logs
'''
# 配置日志
# 禁用 Werkzeug 默认的日志处理器
logger = logging.getLogger('werkzeug')
logger.handlers[:] = []
logger.setLevel(logging.INFO)

root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.INFO)

app = Flask(__name__, 
    static_folder=root_dir+'/tmp', 
    static_url_path='/tmp')

app.logger.setLevel(logging.INFO) 


min_model = CosyVoice('pretrained_models/CosyVoice-300M', load_jit=True, load_trt=False, fp16=True)
tts_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False, load_vllm=False, fp16=True)

# 加载中闽对应词表
min2man_dict = {}
man2min_dict = {}
with open('asset/bilingual_dict_v1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        mandarin, minnan_char, _ = line.strip().split('\t')
        min2man_dict[minnan_char] = mandarin
        man2min_dict[mandarin] = minnan_char
        
def translate(text, from_lang, to_lang):
    # 读取本地 ArgosTranslate 模型包并安装
    model_dir = Path("pretrained_models/argostranslate/")
    for model_path in model_dir.glob("*.argosmodel"):
        argostranslate.package.install_from_path(model_path)

    translated_text = argostranslate.translate.translate(text.strip().lower(), from_lang, to_lang)
    return translated_text        


# 最长匹配，用于中闽互译
def longest_match_replace(text, word_dict):
        result = []
        i = 0
        max_len = max([len(word) for word in word_dict.keys()], default=1)
        while i < len(text):
            matched = False
            # 从最长可能长度开始检查
            for l in range(min(max_len, len(text)-i), 0, -1):
                substr = text[i:i+l]
                if substr in word_dict:
                    result.append(word_dict[substr])
                    i += l
                    matched = True
                    break
            if not matched:
                result.append(text[i])
                i += 1
        return ''.join(result)

# 获取请求参数
def get_params(req):
    params={
        "text":"",
        "lang":"",
        "reference_audio":None,
        "stream":False,
        "speed":1.0
    }
    # 原始字符串
    params['text'] = req.args.get("text","").strip() or req.form.get("text","").strip()
    
    # 字符串语言代码
    params['lang'] = req.args.get("lang","").strip().lower() or req.form.get("lang","").strip().lower()

    if params['lang'][:2] == 'zh':
        # 兼容 zh-cn zh-tw zh-hk
        params['lang']='zh'
    
    # 要克隆的音色文件    
    params['reference_audio'] = req.args.get("reference_audio",None) or req.form.get("reference_audio",None)
    if params['reference_audio'] is None:
        print('未传入参考音频，将使用默认音色')
        params['reference_audio'] = 'asset/zero_shot_prompt.wav'
    
    return params


def del_tmp_files(tmp_files: list):
    print('正在删除缓存文件...')
    for f in tmp_files:
        if os.path.exists(f):
            print('删除缓存文件:', f)
            os.remove(f)


# 实际批量合成完毕后连接为一个文件
def batch(tts_type,outname,params):
    global min_model,tts_model
    if not shutil.which("ffmpeg"):
        raise Exception('必须安装 ffmpeg')
    
    if not params['reference_audio'] or not os.path.exists(f"{root_dir}/{params['reference_audio']}"):
        raise Exception(f'参考音频未传入或不存在 {params["reference_audio"]}')
    ref_audio=f"{tmp_dir}/-refaudio-{time.time()}.wav" 
    try:
        subprocess.run(["ffmpeg","-hide_banner", "-ignore_unknown","-y","-i",params['reference_audio'],"-ar","16000",ref_audio],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                check=True,
                text=True,
                creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        raise Exception(f'处理参考音频失败:{e}')
    
    # 读取参考音频并识别内容
    try:
        print(f"正在处理参考音频: {ref_audio}")
        prompt_speech_text = ''
        prompt_speech_16k = load_wav(ref_audio, 16000)
        import soundfile
        speech, _ = soundfile.read(ref_audio)
        input_len = len(speech) / 16000  # Calculate input length based on sample rate (16kHz)

        detected_language = params.get('lang', 'auto')  # Default to 'auto' if not provided
        asr_res = asr_model.generate(
                input=speech,
                input_len=input_len,
                language=detected_language, 
                batch_size_s=60)
        if len(asr_res) == 0:
            raise Exception('参考音频识别结果为空，请检查参考音频是否正确')
        elif len(asr_res) == 1:
            prompt_speech_text = asr_res[0]['text']
        else:
            prompt_speech_text = ''.join([i['text'].strip() for i in asr_res])
        print(f"参考音频识别结果: {prompt_speech_text}")
    except Exception as e:
        raise Exception(f'识别参考音频失败:{e}')
    
    
    text=params['text']
    audio_list=[]
    if tts_type=='min_tts':
        if params['lang'] == 'zh':
            text = longest_match_replace(text, man2min_dict)
        elif params['lang'] == 'en':
            text = translate(text, 'en', 'zh')
            text = longest_match_replace(text, man2min_dict)
        else:
            raise Exception(f'不支持的语言类型: {params["lang"]}')
        
        print(f"正在进行语音合成, 合成文本: {text}")    
        for _, j in enumerate(min_model.inference_zero_shot(text, prompt_speech_text, prompt_speech_16k, stream=params['stream'], speed=params['speed'])):
            audio_list.append(j['tts_speech'])
            
    elif tts_type=='en_tts':
        if params['lang'] == 'zh':
            text = translate(text, 'zh', 'en')
        elif params['lang'] == 'en':
            pass
        else:
            raise Exception(f'不支持的语言类型: {params["lang"]}')
        
        print(f"正在进行语音合成, 合成文本: {text}") 
        for _, j in enumerate(tts_model.inference_zero_shot(text, prompt_speech_text, prompt_speech_16k, stream=params['stream'], speed=params['speed'])):
            audio_list.append(j['tts_speech'])

    elif tts_type=='zh_tts':
        if params['lang'] == 'zh':
            pass
        elif params['lang'] == 'en':
            text = translate(text, 'en', 'zh')
        else:
            raise Exception(f'不支持的语言类型: {params["lang"]}')
        
        print(f"正在进行语音合成, 合成文本: {text}") 
        for _, j in enumerate(tts_model.inference_zero_shot(text, prompt_speech_text, prompt_speech_16k, stream=params['stream'], speed=params['speed'])):
            audio_list.append(j['tts_speech'])
            
    audio_data = torch.concat(audio_list, dim=1)
    
    # 根据模型yaml配置设置采样率
    if tts_type=='min_tts':
        torchaudio.save(tmp_dir + '/' + outname, audio_data, 22050, format="wav")    
    else:
        torchaudio.save(tmp_dir + '/' + outname, audio_data, 24000, format="wav")    
    
    print(f"音频文件生成成功：{tmp_dir}/{outname}")
    return tmp_dir + '/' + outname


# 闽南语合成语音
@app.route('/min_tts', methods=['GET', 'POST'])        
def min_tts():       
    try:
        params=get_params(request)
        if not params['text']:
                return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
        
        outname=f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-min')}.wav"
        outname=batch(tts_type='min_tts',outname=outname,params=params)
    except Exception as e:
        print(e)
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        return make_response(jsonify({"code":0,"msg":f''}), 200)
        return send_file(outname, mimetype='audio/x-wav')
    
# 中文合成语音      
@app.route('/zh_tts', methods=['GET', 'POST'])               
def zh_tts():
    try:
        params=get_params(request)
        if not params['text']:
            return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
            
        outname=f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-zh')}.wav"
        outname=batch(tts_type='zh_tts',outname=outname,params=params)
    except Exception as e:
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        return send_file(outname, mimetype='audio/x-wav')

# 英文合成语音  
@app.route('/en_tts', methods=['GET', 'POST'])         
def en_tts():

    try:
        params=get_params(request)
        if not params['text']:
            return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
            
        outname=f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-en')}.wav"
        outname=batch(tts_type='en_tts',outname=outname,params=params)
    except Exception as e:
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        return send_file(outname, mimetype='audio/x-wav')
     
         
if __name__=='__main__':
    host='0.0.0.0'
    port=15532
    print(f'\n启动api:http://{host}:{port}\n')
    try:
        from waitress import serve
    except Exception:
        app.run(host=host, port=port)
    else:
        serve(app,host=host, port=port)
    

'''


## 根据内置角色合成文字

- 接口地址:  /tts 
  
- 单纯将文字合成语音，不进行音色克隆

- 必须设置的参数：
 
 `text`:需要合成语音的文字
 
 `role`: '中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女' 选择一个

- 成功返回:wav音频数据

- 示例代码
```
data={
    "text":"你好啊亲爱的朋友们",
    "reference_audio":"10.wav"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```


## 克隆音色合成闽南语  

- 地址：/min_tts

参考音频发音语言需要为普通话或英语，

- 必须设置参数:

`text`： 需要合成语音的文字

`lang`： 需要合成的文字语言，支持 zh（中文）和 en（英文）

`reference_audio`：需要克隆音色的参考音频

`reference_text`：参考音频对应的文字内容 *参考音频相对于 api.py 的路径，例如引用1.wav，该文件和api.py在同一文件夹内，则填写 `1.wav`*

- 成功返回:wav数据

- 示例代码
```
data={
    "text":"你好啊亲爱的朋友们。",
    "reference_audio":"10.wav",
    "reference_text":"希望你过的比我更好哟。"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```

## 不同语言音色克隆: 

- 地址： /cone

参考音频发音语言和需要合成的文字语言不一致，例如需要根据中文发音的参考音频，将一段英文文本合成为语音。

- 必须设置参数:

`text`： 需要合成语音的文字

`reference_audio`：需要克隆音色的参考音频 *参考音频相对于 api.py 的路径，例如引用1.wav，该文件和api.py在同一文件夹内，则填写 `1.wav`*

- 成功返回:wav数据


- 示例代码
```
data={
    "text":"親友からの誕生日プレゼントを遠くから受け取り、思いがけないサプライズと深い祝福に、私の心は甘い喜びで満たされた！。",
    "reference_audio":"10.wav"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```

'''
