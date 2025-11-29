from dotenv import load_dotenv
import os
import numpy as np
import pvporcupine
import pyaudio
import pygame
import wave
import requests
import base64
import uuid


load_dotenv()

PICOVOICE_API_KEY = os.getenv('PICOVOICE_API_KEY')
BACKEND_URL = os.getenv('BACKEND_URL')
TOKEN = os.getenv('TOKEN')

if not PICOVOICE_API_KEY or BACKEND_URL is None:
    raise EnvironmentError('missing keys in environment')#tmp exception handling

porcupine = pvporcupine.create(
    access_key=PICOVOICE_API_KEY,
    keywords=['picovoice','bumblebee']
)


#------- PyAudio ans stream initialization -----------#
FRAME_LENGTH = porcupine.frame_length
SAMPLE_RATE = porcupine.sample_rate 
CHANNELS = 1
FORMAT = pyaudio.paInt16

RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "user_request.wav"

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=FRAME_LENGTH
)


def get_next_audio_frame():
    audio_data_raw = stream.read(FRAME_LENGTH, exception_on_overflow=False)
    audio_frame_pcm = np.frombuffer(audio_data_raw, dtype=np.int16)

    if len(audio_frame_pcm)> porcupine.frame_length:
        print(f"WARNING: expected: {porcupine.frame_length}, Obteined: {len(audio_frame_pcm)}")

    return audio_frame_pcm
#-----------------------------------------------------#


#-------------- User Command Recording ---------------#
def record_user_command(audio_interface,stream):
    frames =  []
    print(f"Recording (max: {RECORD_SECONDS})")
    num_frames = int(SAMPLE_RATE/FRAME_LENGTH * RECORD_SECONDS)

    for i in range(0, num_frames):
        data = stream.read(FRAME_LENGTH,exception_on_overflow=False)
        frames.append(data)
    print(f"Stopped Recording")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return WAVE_OUTPUT_FILENAME
#-----------------------------------------------------#

#-----------------------------------------------------#
def play_audio(audio_bytes):
    tmp_filename = str(uuid.uuid4())+'.wav'
    try:
        with open(tmp_filename,'wb') as f:
            f.write(audio_bytes)
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(1)
        pygame.mixer.music.unload()
    except Exception as e:
        print (f"Errore di riproduzione: {e}")
    finally:
        if os.path.exists(tmp_filename):
            print('trying to remove file')
            os.remove(tmp_filename)
            print('file removed')
#-----------------------------------------------------#

#------------- Sending Audio to Backend --------------#
def send_audio_to_backend(filepath):
    print(f"Sending {filepath} to {BACKEND_URL}")
    try:
        with open(filepath,'rb') as f:
            files = {'audio_file':(filepath,f,'audio/wav')}

            response = requests.post(f"{BACKEND_URL}process_audio",headers={"X-Device-Token":TOKEN},files=files)


            print ("Request submitted successfully")

            return response.json()

    except requests.exceptions.ConnectionError:
        print (f"Connection Error. Unable top reach backend at {BACKEND_URL}")
        return None
#-----------------------------------------------------#


#---------Infinite Loop to look for wake word---------#
while True:
    try:
        #Wake word detection
        audio_frame = get_next_audio_frame()
        keyword_index = porcupine.process(audio_frame)
        if keyword_index >= 0:
            print('---WAKE WORD DETECTED! Recording...---')

            #Record request
            recorded_file = record_user_command(audio,stream)

            #Submit request and receives response
            backend_response = send_audio_to_backend(recorded_file)
            
            if backend_response:
                if backend_response['content']:
                    audio_bytes = base64.b64decode(backend_response['content'].encode('utf-8'))

                    play_audio(audio_bytes)

                    if 'id' in backend_response:
                        id = backend_response['id']
                        response = requests.post(f"{BACKEND_URL}play",headers={"X-Device-Token":TOKEN},json={'id':id})
                        data = response.json()
                        
                        print(data['success'])

                print("Listening...")
    except KeyboardInterrupt:
        print("Closing program...")
        break
    except Exception as e:
        print(f"Critical error: {e}")
        
#-----------------------------------------------------#


#-------------------  Cleanup  -----------------------#
stream.stop_stream()
stream.close()
audio.terminate()
porcupine.delete()
#-----------------------------------------------------#