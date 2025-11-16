import redis
import os
import json
import uuid
from typing import Dict, Optional, Any

REDIS_HOST = os.getenv('REDIS_HOST','localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT',6379))
SESSION_EXPIRATION_SECONDS = 300


class RedisManager:

    def __init__(self):

        self._client: Optional[redis.StrictRedis] = None
        self._fallback_storage: Optional[Dict[str, Dict[str, Any]]] = None
           
        try:
            client = redis.StrictRedis(host=REDIS_HOST,port=REDIS_PORT, db=0, decode_responses = True)
            client.ping()
            self._client = client
            print(f"Redis_Manager: Connessione a Redis stabilita su {REDIS_PORT}")
        except Exception as e:
            print(f"Redis_Manager:ERRORE! Impossibile connettersi a Redis: {e}")
            self._fallback_storage = {}


    def save_session_state(self,data:dict):
        session_id = str(uuid.uuid4())
        if self._client:
            json_data = json.dumps(data)
            self._client.set(session_id,json_data,ex=SESSION_EXPIRATION_SECONDS)
        elif self._fallback_storage is not None:
            self._fallback_storage[session_id]=data
        
        else:
            print("Errore: Redis_Manager non inizializzato correttamente")
            return None
        return session_id
            
    def pop_session(self,session_id)-> Optional[dict]:

        if self._client:
            pipe = self._client.pipeline()
            pipe.get(session_id)
            pipe.delete(session_id)
            result = pipe.execute()

            raw_data = result[0]

            if raw_data:
                return json.loads(raw_data)
            
            return None
        
        elif self._fallback_storage:
            return self._fallback_storage.pop(session_id,None)
        
        else:
            print("Errore: Redis_Manager non inizializzato correttamente")
            return None
        

redis_manager_instance: Optional[RedisManager] = None

def initialize_redis_manager():
    global redis_manager_instance

    if redis_manager_instance is None:
        redis_manager_instance = RedisManager()
    return redis_manager_instance
