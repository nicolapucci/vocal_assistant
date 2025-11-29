import os
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    render_template_string,
    jsonify,
    g
)
import logging
import requests
from clients.AuthHandler import AuthHandler
import json
import uuid
import threading

from clients.spotify_client import SpotifyClient
from managers.redis_manager import initialize_redis_manager
from managers.postgres_manager import initialize_postgres_manager
from decorators import device_endpoint
from tasks.last_fm import LastFm

app = Flask(__name__)
cwd = os.path.dirname(os.path.abspath(__file__))

app.logger.setLevel(logging.DEBUG)

load_dotenv()

SCOPE = os.getenv('SCOPE')

SESSION_STORAGE = {}

BASE_URL = 'http://host.docker.internal'



UPLOAD_FOLDER = 'tmp_audio_uploads'
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

PLAY_MUSIC_INTENTS = [
    'play_music',
    'play_radio',
    'play_podcast',
    'play_audiobook'
]


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

lastFm_manager = LastFm()
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



@app.route('/process_audio',methods=['POST'])#il mio edge device manda qui l'audio registrato
@device_endpoint
def process_audio():
    
    device = g.__getattr__('device')


    user = postgres_manager.get_device_owner(device_id=device.id)

    if 'audio_file' not in request.files:
        audio_response = "audio file missing"
        print('no file')
        return jsonify({"content":audio_response}), 400

    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        print('no filename')
        audio_response = speechProcessor.tts("audio filename missing")
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

        except Exception as e:
            print(f"Eccezione: {e}")
            audio_response = speechProcessor.tts("Eccezione")
            return jsonify({"content": audio_response}), 500
        
        finally:
            os.remove(filepath)

    intent, slots = nluProcessor.process_text(text_input)


    app.logger.info(f"message_tanslation: {text_input}--intent: {intent}--slots:{slots}")

    if intent in PLAY_MUSIC_INTENTS:

        response = spotfy_client.search_spotify_library(user,slots)       

        if response and 'tracks' in response and 'items' in response['tracks']: #essendo una struttura if/else si fermerà solo al primo match. rivedere la priorità.
            type = 'tracks'
        elif response and 'artists' in response and 'items' in response['artists']:
            type = 'artists'
        elif response and 'albums' in response and 'items' in response['albums']:
            type = 'albums'
        elif response and 'playlists' in response and 'items' in response['playlists']:
            type = 'playlists'
        else:
            type = None

        if type is not None:
            items = response[type]['items']#do per scontato che se items è presente sia un array, forse voglio aaiungere un check qua
        else:
            items = None

        #inizializzo tutto a None, se nessuno di questi viene cambiato allora la return sarà 'Riproduco Spotify' e la play ha un modo suo per gestire questo caso
        context_uri = None
        item_name = None
        uri = None


        if type is not None:
            item = items[0]
            item_name = item['name']
            artists = item['artists']
            main_artist = artists[0].get('name')
            if type=='tracks':
                uri = item['uri']
                app.logger.info('sto per chiamare la funzione di refill')
                refill_thread = threading.Thread(
                    target=lastFm_manager.refill_spotify_queue,
                    args=(user,item_name,main_artist)#<-- i need to populate queue only in case the user specified a track, in the other scenarios spotify will handle the queue
                )
                refill_thread.run()
            else:
                context_uri = item['uri']

        app.logger.info(f"type : {type}")
        spotify_device = slots['device'] if 'device' in slots else device#se non è specificato uso chi ha fatto la richiesta

        session_id = redis_manager.save_session_state({'uri':uri,'context_uri':context_uri,'device_id':spotify_device.id})

        audio_response = speechProcessor.tts(f"Riproduco {item_name if item_name is not None else 'Spotify'}")
        return jsonify({ 'content':audio_response, 'id':session_id}),200#tts    

    elif intent == 'unknown':
        audio_response = speechProcessor.tts('Non ho capito.')
        return jsonify({'content':audio_response}),400
    else:
        audio_response = speechProcessor.tts("Quell'azione non è ancora supportata")
        return jsonify({'content':audio_response}),400
    

    

@app.route('/play',methods=['POST'])
@device_endpoint
def play():
    data = request.get_json()
    if not data or 'id' not in data:
        audio_response = speechProcessor.tts("C'è stato un errore")
        return jsonify({'content':audio_response}),400
    id = data['id']
    session = redis_manager.pop_session(session_id=id)

    device = None
    if session['device_id']:
        device = postgres_manager.get_device_by_id(session['device_id'])

    if device is None:#se l'user non specifica il device, o il device specificato non è stato trovato usa ik device che ha fatto la request
        device = g.device

    user = postgres_manager.get_device_owner(device_id=device.id)#questo verrà aggiornato quando aggiorno SpotifyClient, per ora lasciamolo così
    if session is not None:
        try:
            status = spotfy_client.play(user=user,uri=session['uri'],context_uri=session['context_uri'],device_name=None)#ignora il device per ora, manca un mapper tra il db id e lo spotify id per i devices
            return jsonify({'success':status}),200#niente tts se ha successo.
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
