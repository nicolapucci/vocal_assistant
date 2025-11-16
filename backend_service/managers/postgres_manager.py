import os
from typing import Optional, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

#Models
from managers.DB_Models import Base
from managers.DB_Models.User import User
from managers.DB_Models.Device.Device import Device 
from managers.DB_Models.AppSetting.AppSetting import AppSetting
from managers.DB_Models.AppSetting.SpotifySetting import SpotifySetting

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
    def get_device_by_token(self,token:str) -> Optional[dict[str,Any]]:
        session = self._get_session()
        try:
            device: Device = session.query(Device).filter(Device.device_token==token).one_or_none()

            if not device:
                return None
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
        finally:
            session.close()


    def update_app_access_token(self,user:User,updates:dict,app_name:str):
        session = self._get_session()
        try:
            appSetting = session.query(AppSetting).filter(AppSetting.user==user,AppSetting.app_name==app_name).one_or_none()
            
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