import os
from typing import Optional, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

#Models
from managers.DB_Models import Base
from managers.DB_Models.User import User
from managers.DB_Models.Devices.Device import Device 
from managers.DB_Models.AppSettings.AppSetting import AppSetting
from managers.DB_Models.AppSettings.SpotifySetting import SpotifySetting

DATABASE_URL = os.getenv('DATABASE_URL',None)#docker compose lo passa come envirenment

class PostgresManager:

    def __init__(self):
        self.db_url = DATABASE_URL
        self.engine = None
        self.Session = None
        self._is_ready = False

        try:
            self.engine = create_engine(self.db_url)

            Base.metadata.create_all(self.engine)

            self.Session = sessionmaker(bind=self.engine)
            self._is_ready = True
            print("Postgres_Manager: Connessione a PostgreSQL riuscita.")
        except OperationalError as e:
            print(f"Postgras_Manager: ERRORE! Impossibile connettersi a PostgreSQL")

    def  _get_session(self):
        if not self._is_ready or not self.Session:
            raise RuntimeError("Postgres_Manager non inizializzato correttamente")
        return self.Session()
    


#---Qui metodi pubblici per l'uso del DB---#
    def add_user(self,username, clear_password):
        if not username or clear_password is None:
            return 1 #tmp flag
        new_user = User(username=username)
        new_user.set_password(clear_password=clear_password)
        session = self._get_session()
        try:
            session.add(new_user)
            session.commit()
            return new_user
        except Exception as e:
            print(f"Error creating user: {e}")
            session.rollback()
            return 2 #tmp flag
        finally:
            session.close()

    def add_spotify_setting(self,user:User,access_token:str,refresh_token:str,default_device_id:str=None):
        if not user or not access_token or refresh_token is None:
            return None #tmp flaf
        new_spotify_setting = SpotifySetting(user=user,default_spotify_device_id=default_device_id,access_token=access_token,refresh_token=refresh_token)
        session = self._get_session()
        try:
            session.add(new_spotify_setting)
            session.commit()
            return new_spotify_setting
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
            return None #tmp flag
        finally:
            session.close()

    def add_device(self,user:User,device_name:str,device_token:str):
        if not user or not device_name or not device_token:
            return None #tmp flag
        new_device = Device(user=user,device_name=device_name,device_token=device_token)
        session = self._get_session()
        try:
            session.add(new_device)
            session.commit()
            return new_device
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
            return None#tmp flag
        finally:
            session.close()

    def get_app_access_token(self,user_id,app_name):
        if not user_id or app_name is None:
            return None#tmp flag
        session = self._get_session()
        try:
            app_setting = session.query(AppSetting).filter(AppSetting.user_id==user_id,AppSetting.app_name==app_name).one_or_none()
            access_token = app_setting.access_token
            return access_token
        except Exception as e:
            print(f"Error: {e}")
        finally:
            session.close()

    def get_app_refresh_token(self,user_id,app_name):
        if not user_id or app_name is None:
            return None#tmp flag
        session = self._get_session()
        try:
            app_setting = session.query(AppSetting).filter(AppSetting.user_id==user_id,AppSetting.app_name==app_name).one_or_none()
            refresh_token = app_setting.refresh_token
            return refresh_token
        except Exception as e:
            print(f"Error: {e}")
        finally:
            session.close()
        
        

    def get_user_by_username(self,username:str):
        if not username:
            return None
        session = self._get_session()
        try:
            user:User = session.query(User).filter(User.username==username).one_or_none()
            return user
        except Exception as e:
            return None
        finally:
            session.close()

    def get_device_owner(self,device_id:str):
        if not device_id:
            return None #tmp flag
        session = self._get_session()
        try:
            device:Device = session.query(Device).filter(Device.id==device_id).one_or_none()
            user:User = device.user
            return user
        except Exception as e:
            return None#tmp flag
        finally:
            session.close()


    def get_device_by_token(self,token:str) -> Optional[dict[str,Any]]:
        session = self._get_session()
        try:
            device: Device = session.query(Device).filter(Device.device_token==token).one_or_none()
            return device
        except Exception as e:
            print(f"Errore nell'esecuzione della query: {e}")
        finally:
            session.close()

    def get_device_by_id(self,id:int)-> Optional[dict[str,Any]]:
        session = self._get_session()
        try:
            device: Device = session.query(Device).filter(Device.id==id).one_or_none()

            if not device:
                return None
            return device
        except Exception as e:
            print(f"Errore nell'esecuzione della query: {e}")
            return None
        finally:
            session.close()

    def get_device_by_name(self,name:str)-> Optional[dict[str,any]]:
        session = self._get_session()
        try:
            device:Device = session.query(Device).filter(Device.name==name).one_or_none()

            if not device:
                return None
            return device
        except Exception as e:
            print(f"Errore nell'esecuzione della query: {e}")
            return None
        finally:
            session.close()

    def update_app_access_token(self,user:User,updates:dict,app_name:str):
        session = self._get_session()
        try:
            appSetting = session.query(AppSetting).filter(AppSetting.user_id==user.id,AppSetting.app_name==app_name).one_or_none()
            
            if not appSetting:
                return False
            
            if 'access_token' not in updates or updates.get('access_token') is None:
                return False            
            if hasattr(appSetting, 'access_token'):#prima di dargli l'access token mi assicuro che sia uno degi AppSetting che usa l'access token
                appSetting.access_token = updates.get('access_token')

            if hasattr(appSetting,'refresh_token'):
                #se mi da access_token ma non refresh_token mantengo il vecchio refresh
                if 'refresh_token'in updates and updates.get('refresh_token') is not None:
                    appSetting.refresh_token = updates.get('refresh_token')  
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Errore nell'aggiornamento del token: {e}")
            return False
        finally:
            session.close()


#------------------------------------------#

postgres_manager_istance: Optional['PostgresManager'] = None

def initialize_postgres_manager()-> 'PostgresManager':
    global postgres_manager_istance
    if postgres_manager_istance is None:
        postgres_manager_istance = PostgresManager()
    return postgres_manager_istance