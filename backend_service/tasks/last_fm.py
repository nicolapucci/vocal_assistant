from flask import json
import requests
import os

from managers.DB_Models import User
from managers.postgres_manager import initialize_postgres_manager

from clients.spotify_client import SpotifyClient


LAST_FM_URL = 'https://ws.audioscrobbler.com/2.0/'
API_KEY = os.getenv('LAST_FM_KEY')

class LastFm:
    def __init__(self):
        self.auth_handler = AuthHandler()
        self.spotify_client = SpotifyClient()
        self.postgres_manager = initialize_postgres_manager()


    def refill_spotify_queue(self,user:User,song_name:str):#<-- i don't need this for artist/albums/playlists.

        if not song_name:
            return None #tmp flag
        
        params = {
            "method":"track.getsimilar",
            "track":song_name,
            "format":"json",
            "api_key":API_KEY
            }

        
        results = requests.get(#<-- no need to overcomplicate thigs, this is a simple get where the tokes in in the params
            url=LAST_FM_URL,
            params=params
        )
        body = results.json()

        similartracks_data = body.get('similartracks')
        if similartracks_data is None:
            return 

        data = similartracks_data.get('track')
        if data is None:
            return
        
        tracks = []
        for entry in data:
            artist_name = entry.get('artist',{}).get('name','')
            tracks.append({"track":entry['name'],"artist":artist_name})

        for track in tracks:
            if track['name'] != song_name: #<-- i already added song_name to the queue b4 looking for similar songs
                try:
                    #when i'll add caching i will check the cache here.
                    search_params = {'song_name':track['name'],'artist_name':track['artist']}
                    spotify_response = self.spotify_client.search_spotify_library(user=user,search_params=search_params)

                    uri =  spotify_response['tracks']['items'][0]['uri']
                    self.spotify_client.add_to_queue(user,uri)
                    #then add it to the cache

                except Exception as e:
                    print('dfrojnsgfz')#to be replaced with logging
                
        return