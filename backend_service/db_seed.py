import os 
from managers.postgres_manager import initialize_postgres_manager
from dotenv import load_dotenv


load_dotenv()

username = os.environ.get('SEED_USERNAME')
password = os.environ.get('SEED_PASSWORD')
access_token = os.environ.get('SPOTIFY_ACCESS_TOKEN')
refresh_token = os.environ.get('SPOTIFY_REFRESH_TOKEN')
device_token = os.environ.get('EDGE_DEVICE_TOKEN')

print(username,password)

postgres_manager = initialize_postgres_manager()
user = postgres_manager.get_user_by_username(username=username)
if user is None:
    print('creating user')
    user = postgres_manager.add_user(username=username,clear_password=password)
    print(user)
if user in [1,2]:
    exit()
device = postgres_manager.get_device_by_token(device_token)
if not device:
    print('creating device')
    device = postgres_manager.add_device(
        user=user,
        device_name='edge_device',
        device_token=device_token,
    )

postgres_manager.add_spotify_setting(
    user=user,
    access_token=access_token,
    refresh_token=refresh_token
)