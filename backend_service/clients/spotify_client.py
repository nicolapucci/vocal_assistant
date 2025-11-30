import requests
from urllib.parse import urlencode
import base64
import os
import threading

import logging

from clients.AuthHandler import AuthHandler

from managers.redis_manager import initialize_redis_manager
from managers.postgres_manager import initialize_postgres_manager

SPOTIFY_URL = 'https://accounts.spotify.com/'
SLOT_MAPPING = {
        'artist_name': ('artist', 'artist'),
        'song_name': ('track', 'track'),
        'album_name': ('album', 'album'),
    }

logger = logging.getLogger(__name__)
redis_manager = initialize_redis_manager()
postgres_manager = initialize_postgres_manager()


CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')
SCOPE = os.getenv('SCOPE')


class SpotifyClient:

    def  __init__(self):
        self.authHandler = AuthHandler()

#-----------------------------
#       TOOLS
#-----------------------------
    @staticmethod
    def build_header_token_only(token:str):
        return {'Authorization': f"Bearer {token}"}
    
    @staticmethod
    def build_header_token_content(token:str):
            return {'Authorization': f"Bearer {token}",
                    'Content-Type':'application/json'
                    }

    @staticmethod
    def build_auth_url(username):
        return f'{SPOTIFY_URL}authorize?' + urlencode({
            'response_type': 'code',
            'client_id': CLIENT_ID,
            'scope': SCOPE,
            'redirect_uri': REDIRECT_URI,
            'state': username
        })

    @staticmethod
    def exchange_code_for_tokens(auth_code):
        auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

        token_data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': REDIRECT_URI
        }

        headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.post(
            'https://accounts.spotify.com/api/token',
            data=token_data,
            headers=headers
        )
        
        return response.json()
    
    @staticmethod
    def build_refresh_token_call(refresh_token):
        url = "https://accounts.spotify.com/api/token"

        auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        headers = {
            'Authorization': f'Basic {auth_header}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data={
            "grant_type":"refresh_token",
            "refresh_token":refresh_token
            }
        return url,headers,data
    
    @staticmethod
    def build_play_track_call(access_token,track_uri):
        url = 'https://api.spotify.com/v1/me/player/play'
        headers={
            'Authorization': f"Bearer {access_token}",
            'Content-Type':'application/json'
                }
        json={
                'uris':[track_uri],
                'position_ms':0
            }
        return url,headers,json

#-----------------------------
#       UTILITIES
#-----------------------------
    def refresh_spotify_access_token(self,user):
        success = self.authHandler.refresh_token(
            user=user,
            app_name='spotify',
            build_refresh_call=self.build_refresh_token_call
        )

        return success       

    def map_devices(self,user):
        url = 'https://api.spotify.com/v1/me/player/devices'

        def build_header(token):
            return {'Authorization': f"Bearer {token}"}
        try:
            response = self.authHandler.apiPrivate(
                user=user,
                build_header=build_header,
                url=url,
                method='GET',
                refresh_call=self.refresh_spotify_access_token,
                app='spotify'
            )

            if 'devices' in response:
                devices = response['devices']
                return devices
            
        except Exception as e:
            print(f"Error fetching devices data: {e}")
        return False #tmp flag

    def search_spotify_library(self,user, search_params): 
        url = 'https://api.spotify.com/v1/search'

        
        search_query_parts = []
        spotify_types = set() 
        
        for slot in search_params:
            slot_keys = slot.keys()
            for slot_key in slot_keys:
                
                if slot_key in SLOT_MAPPING:
                    query_prefix, spotify_type = SLOT_MAPPING[slot_key]
                    
                    search_query_parts.append(f"{query_prefix}:{slot[slot_key]}")
                    
                    spotify_types.add(spotify_type)

        if not search_query_parts:
            return None
        
        final_query = " ".join(search_query_parts)
        final_types = ",".join(spotify_types)

        params={
                'q': final_query, 
                'type': final_types, 
                'limit': 1 
            }

        try:
            response = self.authHandler.apiPrivate(
                user=user,
                build_header=self.build_header_token_only,
                url=url,
                params=params,
                method='GET',
                app='spotify',
                refresh_call=self.refresh_spotify_access_token
            )

            return response
        
        except Exception as e:
            logger.exception(f"ERRORE! C'è stato un errore nella chiamata: {e}")
            return None

    def add_to_queue(self,user,uri):#<-- spotify only allows to add 1 song each time
        url = "https://api.spotify.com/v1/me/player/queue"

        if not uri or user is None:
            return

        try:
            self.authHandler.apiPrivate(
                user=user,
                build_header=self.build_header_token_only,
                url=url,
                method='POST',
                app='spotify',
                refresh_call=self.refresh_spotify_access_token,
                params={"uri":uri}
            )
            return
        except Exception as e:
            #logging here
            return
#-----------------------------
#     GENERATE PLAY CONTEXT
#-----------------------------
    def generate_spotify_context(self,user,device,slots):

        response = self.search_spotify_library(user,slots)       
    
        #Voglio prendo il tipo dell'oggetto da estrarre, filtrando per priorità (track->album->artist->playlist->None), controllo anche che non siano oggetti vuoti
        if response and 'tracks' in response and 'items' in response['tracks']:
            type = 'tracks'
        elif response and 'albums' in response and 'items' in response['albums']:
            type = 'albums'
        elif response and 'artists' in response and 'items' in response['artists']:
            type = 'artists'
        elif response and 'playlists' in response and 'items' in response['playlists']:
            type = 'playlists'
        else:
            type = None #fallback, verrà ripreso il playback presente

        if type is not None:
            items = response[type]['items']#nel check precedente ho già controllato che esiste
        else:
            items = None

        #inizializzo le possibili opzioni
        context_uri = None
        item_name = None
        uris = []


        if type is not None and len(items)>0 :#se ho il tipo e items non è un array vuoto
            item = items[0]
            item_name = item['name']#<-- tutti i type hanno un attributo name
            if type == 'tracks':
                artists = item['artists']
                artist = artists[0].get('name') if len(artists)>0 and 'name' in artists[0] else None#nel caso sia una track mi serve il nome dell'artista per la ricerca su last.fm track.getsimilar

                uris.append(item['uri'])#questo deve essere una lista di stringhe dove ogni stringa rappresenta una track
            else:
                context_uri = item['uri']#questo deve essere una stringa rappresentante un artista/album/playlist

        spotify_device = slots['device'] if 'device' in slots else device#se device non è specificato negli slot uso il device che ha fatto la richiesta

        session_id = redis_manager.save_session_state({'uris':uris,'context_uri':context_uri,'device_id':spotify_device.id,'track':item_name,'artist':artist})#salvo una sessione con tutte le informazioni pronte per spotify me/player/play
        should_call_lastFm = uris is not None and len(uris)>0 
        return session_id,item_name,should_call_lastFm#session_id servirà per chiamare backend/play, item_name serve per generare una risposta all'user (eg. riproduco Diamond Eyes di Shinedown)
    
#-----------------------------
#START PLAYBACK (uses context)
#-----------------------------
    def play(self,session_id,user):
        if not user or session_id is None:
            return#tmp flag
        
        session = redis_manager.pop_session(session_id=session_id)#estraggo i dati necessari dalla sessione
        uris = session.get('uris',None)
        context_uri = session.get('context_uri',None)
        device_id = session.get('device_id',None)

        if device_id is not None:
            try:
                specified_device = postgres_manager.get_device_by_id(device_id)
                device_name = specified_device['name']
            except Exception as e:
                device_name = None 
        else:
            device_name = None

#questo non viene ancora usato correttamente, trovare un metodo efficace per fare mapping dei devices
        devices = self.map_devices(user)
        default_device = device_name if device_name is not None else 'DESKTOP-TF5OKBM'
        device_to_use = None
        if devices:     
            for device in devices:
                if device['name']==default_device:
                    device_to_use=device
#----------------------------------------------------------------------------------------------------

        params = {"device_id":device_to_use['id']}
        url = f'https://api.spotify.com/v1/me/player/play?'
        json_body={'position_ms':0,}

        if uris is not None:
            json_body['uris']=uris
        elif context_uri is not None:
            json_body['context_uri']=context_uri
        else:
            json_body = None #Controllare cosa succede se spotify non è in riproduzione e/o non ha una coda di riproduzione.

        try:
            self.authHandler.apiPrivate(
                    user=user,
                    build_header=self.build_header_token_content,
                    url=url,
                    json_body=json_body,
                    params=params,
                    method='PUT',
                    app='spotify',
                    refresh_call=self.refresh_spotify_access_token
                )
            return True #apiPrivate uses response.raise_for_status() and spotify only returns the status code
        except Exception as e:
            logger.exception(f'Error!: {json_body}')
            return False


