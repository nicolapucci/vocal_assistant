import whisper
import torch
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
import uuid

from TTS.api import TTS
from pathlib import Path
import base64

from flask import (
    Flask,
    request,
    jsonify
)

app= Flask(__name__)

cwd = Path(__file__).parent
tmp = cwd / 'tmp'
PORT = 5001
#torch.serialization.add_safe_globals([XttsConfig,XttsAudioConfig,BaseDatasetConfig,XttsArgs])

class SpeechProcessor:
    def __init__(self, model_name:str = 'medium'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = whisper.load_model(model_name,device=self.device)

        self.text_to_speech = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def stt(self,audio_filepath:str)-> dict:
        if not os.path.exists(audio_filepath):
            return {'text':f"Error:Audio file not found at {audio_filepath}"}
        
        try:
            result = self.model.transcribe(audio_filepath)

            return {
                'text':result.get('text','').strip(),
                'language':result.get('language','')
            }
        except Exception as e:
            return {'text':f"Error during STT elaboration with Whisper: {e}"}
        
    def tts(self,string:str):
        SAMPLE_VOICE_PATH = cwd / 'sample.wav'
        output_filename = 'output_tts.wav'
        output_filepath = tmp / output_filename

        if not os.path.exists(tmp):
            os.makedirs(tmp)

        self.text_to_speech.tts_to_file(
            text=string,
            speaker_wav=str(SAMPLE_VOICE_PATH),
            language='it',
            file_path=str(output_filepath)
        )

        with open(output_filepath,'rb') as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        return encoded_string

speechProcessor = SpeechProcessor()

@app.route('/stt',methods=['POST'])
def stt():
    if 'audio_file' not in request.files:
        audio_response = speechProcessor.stt("manca audio_file")
        return jsonify({'content':audio_response}),400
    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        audio_response = speechProcessor.tts("audio file è \"\"")
        return jsonify({'ok':False,'content':audio_response})
    
    if audio_file:
        unique_filename = str(uuid.uuid4()) +os.path.splitext(audio_file.filename)[1]
        filepath = tmp / unique_filename
        audio_file.save(str(filepath))

        result = speechProcessor.stt(str(filepath))

        if 'Errore' in result.get('text',''):
            audio_response = speechProcessor.tts('Non ho capito')
            return jsonify({'ok':False,'content':audio_response})
        return jsonify({'ok':True,'content':result['text']})

@app.route('/tts',methods=['POST'])
def tts():
    if not request.is_json or 'text' not in request.json:
        audio_response = speechProcessor.stt("Manca text")
        return jsonify({'ok':False,'content':audio_response}),400
    
    text = request.json['text']
    if text:
        audio_response = speechProcessor.tts(text)
        return jsonify({'ok':True,'content':audio_response})

    audio_response = speechProcessor.tts("return di fallback")
    return jsonify({'ok':False,'content':audio_response})

if __name__ == '__main__':
    print(f"Avvio del server di configurazione su http://localhost:{PORT}")
    app.run(port=PORT, debug=True)