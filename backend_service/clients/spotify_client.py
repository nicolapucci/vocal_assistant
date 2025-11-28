import requests
import json
from urllib.parse import urlencode
import base64
import os

from clients.AuthHandler import AuthHandler
from managers.DB_Models.Device import Device
from managers.postgres_manager import initialize_postgres_manager

SPOTIFY_URL = 'https://accounts.spotify.com/'
SLOT_MAPPING = {
        'artist_name': ('artist', 'artist'),
        'song_name': ('track', 'track'),
        'album_name': ('album', 'album'),
    }



CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')
SCOPE = os.getenv('SCOPE')

postgras_manager = initialize_postgres_manager()

class SpotifyClient:

    def  __init__(self):
        self.authHandler = AuthHandler()

#---------------------------
#       UTILS
#---------------------------
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


    def refresh_spotify_access_token(self,user):
        success = self.authHandler.refresh_token(
            user=user,
            app_name='spotify',
            build_refresh_call=self.build_refresh_token_call
        )

        return success       

#---------------------------
#     SEARCH ON SPOTIFY
#---------------------------
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
            return {"error": "Nessuno slot di ricerca valido trovato."}
        
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
            response.raise_for_status()
        
            return response
        
        except Exception as e:
            print(f"ERRORE! C'è stato un errore nella chiamata")
            return None


    def generate_radio(self,user,ids:list, category:str):
        url='https://api.spotify.com/v1/reccomendations'
        
        
        if len(uris) == 0:
            return None
        elif len(uris) > 5:
            uris = uris[:5]
            
        params = {'limit':50}
        if category == 'artists':
            params['seed_artists']=ids
        elif category == 'genres':
            params['seed_genres'] = ids
        else: #fallback, if it doesn't match all the other categories with tracks
            params['seed_tracks'] = ids
            
            
        def build_header(token:str):
            return {'Authorization': f'Bearer {token}'}
        
        try:
            response = self.authHandler.apiPrivate(
                user=user,
                build_header=self.build_header_token_only,
                url=url,
                method='GET',
                app='spotify',
                refresh_call=self.refresh_spotify_access_token,
                params=params
            )
            
            response.raise_for_status()
            
            tracks = response['tracks'] if 'tracks' in response else None

            uris = []

            if tracks is not None: #i only return the uris.
                for track in tracks:
                    uris.append(track['uri'])

            return uris
        except Exception as e:
            print(f"Errore nella raccolta dei dati: {e}")
            return None
            
            
#---------------------------
#     PLAY SPOTIFY
#---------------------------
    def play(self,user,uris:list=None,context_uri:str=None,device_name:str=None):
        
        devices = self.map_devices(user)
            
        default_device = device_name if device_name is not None else 'DESKTOP-TF5OKBM'

        device_to_use = None

        #for param in search_params:
            #if 'device_type' in param.keys():#if device is specified use the specified device
                #for device in devices:             
                    #if device['name']==search_params['device_type']:
                        #device_to_use = device
                    
        for device in devices:
            if device['name']==default_device:
                device_to_use=device
                
        param_device = f'device_id={device_to_use["id"]}' if device_to_use is not None else ''

        params = {
            "device_id":device_to_use_api['id']
        }

        url = f'https://api.spotify.com/v1/me/player/play?{param_device}'



        def build_header(token):
            return {'Authorization': f"Bearer {token}",
                    'Content-Type':'application/json'
                    }
        json_body={
                'position_ms':0,
            }
        if uris is not None and len(uris)>0:
            json_body['uris']=uris
        elif context_uri is not None:
            json_body['context_uri']=context_uri
        else:
            json_body = None #Controllare cosa succede se spotify non è in riproduzione e/o non ha una coda di riproduzione.




#---------------------------
#     DEVICE MAPPER
#---------------------------
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
