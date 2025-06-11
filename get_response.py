import requests

url = 'http://127.0.0.1:15532/min_tts'
data={
    "text":"Hello, this is a test of the TTS system.",
    "reference_audio":"10.wav",
    "lang":"en"
}
response = requests.post(url, data=data, timeout=3600)

# 验证响应
if response.status_code == 200 and response.headers.get('Content-Type') == 'audio/x-wav':
    print('测试通过！返回音频文件')
    with open('test_tts.wav', 'wb') as f:
        f.write(response.content)
else:
    print(f'测试失败！状态码：{response.status_code}，响应内容：{response.text}')