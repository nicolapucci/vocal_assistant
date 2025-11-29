from flask import json
import requests
import os
import logging
import re
import time

from managers.DB_Models import User

from clients.spotify_client import SpotifyClient


LAST_FM_URL = 'https://ws.audioscrobbler.com/2.0/'
API_KEY = os.getenv('LAST_FM_KEY')
DELAY_SECONDS = 2

def normalize_track_name(name: str) -> str:
    name = re.sub(r'\s*\(.*\)', '', name)
    name = re.sub(r'\s*\[.*\]', '', name)
    name = name.strip()
    return name


logger = logging.getLogger(__name__)

class LastFm:
    def __init__(self):
        self.spotify_client = SpotifyClient()


    def refill_spotify_queue(self,user:User,song_name:str,artist_name:str):#<-- i don't need this for artist/albums/playlists.

        if not song_name:
            logger.error('No song name provided')
            return None #tmp flag
        
        params = {
            "method":"track.getsimilar",
            "track":normalize_track_name(song_name),
            "artist":normalize_track_name(artist_name),
            "format":"json",
            "api_key":API_KEY,
            "autocorrect":1,
            "limit":50
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

        time.sleep(DELAY_SECONDS)
        for track in tracks:
            if track['name'] != song_name: #<-- i already added song_name to the queue b4 looking for similar songs
                try:
                    #when i'll add caching i will check the cache here.
                    search_params = [{'song_name':track['name']},{'artist_name':track['artist']}]
                    spotify_response = self.spotify_client.search_spotify_library(user=user,search_params=search_params)

                    uri =  spotify_response['tracks']['items'][0]['uri']
                    self.spotify_client.add_to_queue(user,uri)
                    #then add it to the cache

                except Exception as e:
                    logger.exception(f"Error in populating the queue; {e}")
                
        return