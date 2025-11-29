import requests
import os
import json
from pathlib import Path
from datetime import datetime,timedelta,timezone

from managers.DB_Models.User import User

from managers.postgres_manager import initialize_postgres_manager

postgres_manager = initialize_postgres_manager()

class AuthHandler:


    def refresh_token(self,user:User, app_name:str, build_refresh_call:callable):#so multiple modules can use this func, and when there will be a database only this will change and the clients will work fine

        
        refresh_token = postgres_manager.get_app_refresh_token(user_id=user.id,app_name=app_name)

        try:
            url,headers,data = build_refresh_call(refresh_token)
            response = requests.post(url=url,headers=headers,data=data)

            response.raise_for_status()

            response_data = response.json()

            new_access_token = response_data.get('access_token')
            expires_in = response_data.get('expires_in',3600)
            new_refresh_token = response_data.get('refresh_token',refresh_token)
            
            if not new_access_token:
                print("Errore Refresh: Il server OAuth non ha restituito un access_token valido.")
                return False

            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(seconds=expires_in)#non previsto nel modello
            updates = {
                'access_token':new_access_token,
                'refresh_token':new_refresh_token
            }

            success = postgres_manager.update_app_access_token(
                user=user,
                updates=updates,
                app_name=app_name)
                    
            return success #True se va tutto bene, False se qualcosa è andato storto e il DB non ha subitop modifiche
        except Exception as e:
            return False#to be handled later / HTTP error


    def apiPrivate(self,
                   user,
                   build_header:callable,
                   url:str,
                   method:str,
                   app:str,
                   refresh_call:callable,
                   json_body=None,
                   params=None
                   ):
        
        
        access_token = postgres_manager.get_app_access_token(user_id=user.id,app_name=app)

        if not access_token:
            return False#tmp debug handling
        
        header = build_header(access_token)

##the following part is just a quick implementetion, need to be reimplemented in a more robust way
        handler = None
        if method == 'POST':
            handler = requests.post
        elif method == 'PUT':
            handler = requests.put
        else:
            handler = requests.get

        response = handler(url,headers=header,params=params,json=json_body)

        if response.status_code == 401:

            is_access_token_refreshed = refresh_call(user)
            if is_access_token_refreshed:

                access_token = postgres_manager.get_app_access_token(user_id=user.id,app_name=app)

                header = build_header(access_token)#rebuild a header with new access_token

                response = handler(url,headers=header,params=params,json=json_body)

                response.raise_for_status()

                if response.content:
                    return response.json()
                else:
                    return response.status_code
            raise Exception #tmp flag
        else:
            response.raise_for_status()#i do it now because i want to check if it's a 401 before throwing an exception

            return response.json()
        
