from flask import Flask, request, jsonify
import torchaudio

app = Flask(__name__)

@app.route('/cosytts', methods=['POST'])
def handle_request():
    # 获取tts文本
    tts_text = request.form.get('tts_text')
    
    # 获取prompt音频
    prompt_speech = request.files['prompt_speech']
    if not prompt_speech:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    return jsonify({
        'result1': tts_text,
        'result2': prompt_speech.filename
    })
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)