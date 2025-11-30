from flask import json
import requests
import os
import logging
import re
import time

from managers.DB_Models import User

from managers.redis_manager import initialize_redis_manager
from clients.spotify_client import SpotifyClient


LAST_FM_URL = 'https://ws.audioscrobbler.com/2.0/'
API_KEY = os.getenv('LAST_FM_KEY')

def normalize_track_name(name: str) -> str:
    name = re.sub(r'\s*\(.*\)', '', name)
    name = re.sub(r'\s*\[.*\]', '', name)
    name = name.strip()
    return name

redis_manager = initialize_redis_manager()
logger = logging.getLogger(__name__)

class LastFm:
    def __init__(self):
        self.spotify_client = SpotifyClient()


    def generate_spotify_radio(self,user:User,session_id:str):#<-- i don't need this for artist/albums/playlists.

        if not session_id:
            logger.error('No session_id provided')
            return None #tmp flag
        
        session = redis_manager.get_session_state(session_id=session_id)

        song_name:str = session.get('track')
        artist_name:str = session.get('artist')
        uris:list = session.get('uris')

        if not song_name or not artist_name or uris is None:
            return #logging here
        
        params = {
            "method":"track.getsimilar",
            "track":normalize_track_name(song_name),
            "artist":normalize_track_name(artist_name),
            "format":"json",
            "api_key":API_KEY,
            "autocorrect":1,
            "limit":15#<--per aumentanre il numero di traccie bisogna ridurre i tempi di esecuzioni(ciclo for con chiamata a /search deve molte chiamate concorrenti)
            }

        results = requests.get(#<-- no need to overcomplicate thigs, this is a simple get where the tokes in in the params
            url=LAST_FM_URL,
            params=params
        )
        body = results.json()


        similartracks_data = body.get('similartracks')
        if similartracks_data is None:
            logger.error('similartracks is none')
            return 

        data = similartracks_data.get('track')
        
        if data is None:
            logger.error('track is none')
            return
        
        tracks = []
        for entry in data:
            artist_name = entry.get('artist',{}).get('name','')
            tracks.append({"name":entry['name'],"artist":artist_name})

        for track in tracks:#<-- creare 1 thread parallelo per ogni track, così da minimizzare i tempi necessari
            if track['name'] != song_name: #<-- i already added song_name to the queue b4 looking for similar songs
                try:
                    #when i'll add caching i will check the cache here.
                    search_params = [{'song_name':track['name']},{'artist_name':track['artist']}]
                    spotify_response = self.spotify_client.search_spotify_library(user=user,search_params=search_params)

                    uri =  spotify_response['tracks']['items'][0]['uri']

                    uris.append(uri)

                except Exception as e:
                    logger.exception(f"Error in populating the queue; {e}")

        redis_manager.save_session_state(session_id=session_id,data=session)
        return