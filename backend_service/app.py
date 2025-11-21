import os
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    render_template_string,
    jsonify,
    g
)
import requests
from clients.AuthHandler import AuthHandler
import json
import uuid

from clients.spotify_client import SpotifyClient
from managers.redis_manager import initialize_redis_manager
from managers.postgres_manager import initialize_postgres_manager
from decorators import device_endpoint

app = Flask(__name__)
cwd = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

SCOPE = os.getenv('SCOPE')

SESSION_STORAGE = {}

BASE_URL = 'http://127.0.0.1'



UPLOAD_FOLDER = 'tmp_audio_uploads'
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


#----Interfaccia di comunicazione con i modelli stt tts e nlu----#
class SpeechProcessor:
    @staticmethod
    def tts(text:str):
        try:
            ttsResponse = requests.post(
                f"{BASE_URL}:5001/tts",
                json={'text':text}
            )
            ttsResponse.raise_for_status()

            data = ttsResponse.json()
            binary_audio = data['content']#tts ritorna un binario base 64 che l'edge device decodifica in audio
            return binary_audio
        except Exception as e:
            print(f"Errore di comunicazione con lo speechProcessor: {e}")
            return {'ok':False,'content':'Errore di Comunicazione con speech processor'}

    @staticmethod
    def stt(filepath):
        try:
            with open(filepath,'rb') as f:
                files = {'audio_file':(filepath,f,'audio/wav')}
                sttResponse = requests.post(
                    f"{BASE_URL}:5001/stt",
                    files=files
                )
                sttResponse.raise_for_status()

                data = sttResponse.json()
                text = data['content']
                return text
        except Exception as e:
            print(f"Errore di comunicazione con lo speechProcessor: {e}")
            return {'ok':False,'content':'Errore di Comunicazione con speech processor'}

class NLUProcessor:

    @staticmethod
    def process_text(text:str):
        try:
            nluResponse = requests.post(
                f"{BASE_URL}:5002",
                json={'text':text},
            )
            nluResponse.raise_for_status()

            data = nluResponse.json()
            intent = data['content']['intent']
            slots = data['content']['slots']
            return intent,slots
        except Exception as e:
            print(f"Errore di comunicazione con nluProcessor: {e}")
            return {'ok':False,'content':'Errore di Comunicazione con nlu processor'}

speechProcessor = SpeechProcessor()
nluProcessor = NLUProcessor()
#----------------------------------------------------------------#


#----------------------------Clients-----------------------------#
spotfy_client = SpotifyClient()
authHandler = AuthHandler()
#additional clients here
#----------------------------------------------------------------#



#----------------------------Managers----------------------------#
redis_manager = initialize_redis_manager()
postgres_manager = initialize_postgres_manager()
#----------------------------------------------------------------#




#-----------------------------Routes-----------------------------#
@app.route('/')
def home():
    """Pagina principale: mostra un elenco di utenti e un modo per aggiungerne uno."""
    user_data = authHandler.load_user_data()
    
    new_user_id = "user_" + str(len(user_data) + 1).zfill(2)

    nicola_id = "nicola_home"
    nicola_auth_link = spotfy_client.build_auth_url(new_user_id)
    
    html = f"""
    <h1>Dashboard di Configurazione Vocal Assistant</h1>
    <h2>1. Utenti Registrati</h2>
    <pre>{json.dumps(user_data, indent=4)}</pre>
    
    <h2>2. Aggiungi/Riconnetti Utente</h2>
    <p>Clicca sul link qui sotto per avviare l'autenticazione Spotify per l'utente **{nicola_id}**.</p>
    
    <a href="{nicola_auth_link}">Clicca qui per autenticare/aggiornare i token di NICOLA</a>
    <hr>
    
    <p>In futuro, il tuo frontend chiederà l'ID utente e genererà un link come questo per l'autenticazione.</p>
    """
    return render_template_string(html)

@app.route('/callback')
def callback():
    """Gestisce il reindirizzamento da Spotify."""
    
    auth_code = request.args.get('code')
    user_id = request.args.get('state') 
    error = request.args.get('error')

    if error:
        return f"❌ Errore di Autorizzazione per {user_id}: {error}"

    if not auth_code or not user_id:
        return "❌ Errore: parametri mancanti nel callback."

    token_info = spotfy_client.exchange_code_for_tokens(auth_code)

    if 'error' in token_info:
        return f"❌ Errore nello scambio dei token: {token_info.get('error_description', token_info['error'])}"
    
    access_token = token_info.get('access_token')
    refresh_token = token_info.get('refresh_token')

    user_data = authHandler.load_user_data()
    
    if user_id not in user_data:
        user_data[user_id] = {'name': user_id, 'apps': {}}
        
    user_data[user_id]['apps']['spotify'] = {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'scope': token_info.get('scope', SCOPE),
        'expires_in': token_info.get('expires_in', 3600)
    }

    authHandler.save_user_data(user_data)
    
    return f"""
    <h1>✅ Successo!</h1>
    <p>I token Spotify per l'utente **{user_id}** sono stati aggiornati e salvati.</p>
    <p>Ora puoi chiudere questa finestra o tornare alla <a href="/">Dashboard</a>.</p>
    <pre>Refresh Token salvato: {refresh_token[:10]}...</pre>
    """

@device_endpoint
@app.route('/process_audio',methods=['POST'])#il mio edge device manda qui l'audio registrato
def process_audio():

    device = g.device

    user = device.user#ora il decorator mi garantisce che device.user esista

    if 'audio_file' not in request.files:
        audio_response = speechProcessor.tts("audio_file_missing")
        return jsonify({"content":audio_response}), 400

    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        audio_response = speechProcessor.tts("audio_filename_\"\"")
        return jsonify({"content":audio_response}), 400
    
    if audio_file:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(audio_file.filename)[1]
        filepath = os.path.join(cwd,app.config['UPLOAD_FOLDER'], unique_filename)
        print(f'path:{filepath} ')
        try:
            try:
                audio_file.save(str(filepath))
            except Exception as e:
                print(f"Error saving file: {e}")

            result = speechProcessor.stt(filepath)
                 
            print(f"Result: {result}")
            text_input =result

        except Exception as e:#questo viene triggerato
            print(f"Eccezione: {e}")
            audio_response = speechProcessor.tts("Eccezione")
            return jsonify({"content": audio_response}), 500
        finally:
            os.remove(filepath)

    intent, slots = nluProcessor.process_text(text_input)

    print(f"message_tanslation: {text_input}--intent: {intent}--slots:{slots}")

    if intent == 'play_music':

        response = spotfy_client.search_spotify_library(user,slots)
        if 'error' in response:
            audio_response = speechProcessor.tts(response['message'])
            return jsonify({'content':audio_response}),404#tts
        
        track = response['items'][0] if 'items' in response['tracks'] else None

        if track is not None:
            track_uri = track['uri'] if track else None
            track_name =track['name'] if track else None

            session_id = redis_manager.save_session_state({'uris':track_uri})

            audio_response = speechProcessor.tts(f"Riproduco {track_name}")
            return jsonify({ 'content':audio_response, 'id':session_id}),200#tts   
        else:
            audio_response = speechProcessor.tts(f"Non ho trovato canzoni con titolo {track_name}")
            return jsonify({'content':audio_response}),404#? kinda si e kinda no non ha trovato corrispondenza su spotify        

    elif intent == 'unknown':
        audio_response = speechProcessor.tts('Non ho capito.')
        return jsonify({'content':audio_response}),400
    else:
        audio_response = speechProcessor.tts("Quell'azione non è ancora supportata")
        return jsonify({'content':audio_response}),400
    
@device_endpoint
@app.route('/play',methods=['POST'])
def play():
    data = request.get_json()
    if not data or 'id' not in data:
        audio_response = speechProcessor.tts("C'è stato un errore")
        return jsonify({'content':audio_response}),400
    id = data['id']
    session = redis_manager.pop_session(session_id=id)
    device = g.device
    user = device.user#questo verrà aggiornato quando aggiorno SpotifyClient, per ora lasciamolo così
    if session is not None:
        try:
            spotfy_client.play_track(user=user,track_uri=session['uris'])#ignora il device per ora
            return jsonify({'success':True})
        except Exception as e:
            audio_response = speechProcessor.tts(f'Errore nell\'avvio della riproduzione. : {e}')
            return jsonify({'success':False,'content': audio_response}), 500
    else:
        audio_response =speechProcessor.tts(f'Errore nell\'avvio della riproduzione.')
        return jsonify({'success':False,'content': audio_response}), 500
#----------------------------------------------------------------#


if __name__ == '__main__':
    print(f"Avvio del server di configurazione su http://localhost:5000")
    app.run()