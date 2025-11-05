from flask import Flask,request,jsonify
import torch
import os
import wave
import json
import tempfile
from vosk import (Model,
                  KaldiRecognizer)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification)
from datasets import load_dataset
import numpy as np

app = Flask(__name__)
VOSK_MODEL_PATH = "vosk-model-en"

print(f"CUDA disponibile in PyTorch: {torch.cuda.is_available()}")
print(f"GPU Trovata: {torch.cuda.get_device_name(0)}")

#--- load stt model ----#
try:
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {VOSK_MODEL_PATH}")
    
    vosk_model = Model(VOSK_MODEL_PATH)
    print("Model loeaded successfully")

except Exception as e:
    print (f"Critical Error: {e}")
    vosk_model=None
#-----------------------#

def clean_command_text(text):
    # Rimuove apostrofi e li sostituisce con un semplice spazio (o li elimina)
    text = text.replace("'s", "")
    text = text.replace("'", "")
    # Rimuove punteggiatura, ecc., se necessario
    # text = re.sub(r'[^\w\s]', '', text) 
    return text.lower().strip()


#-- stt transcription --#
def transcribe_audio(filepath):
    if vosk_model is None:
        return "Error: vosk model non loaded"
    
    wf = wave.open(filepath,'rb')

    if wf.getnchannels()!= 1 or wf.getsampwidth()!=2:
        print("WARNING: Audio format not Mono or 16-bit")
        return ""
    
    recognizer = KaldiRecognizer(vosk_model,wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)
    
    result_json = recognizer.FinalResult()

    text = json.loads(result_json).get('text','')

    return text.lower()
#-----------------------#



INTENTS = ['datetime_query',
           'iot_hue_lightchange',
           'transport_ticket',
           'takeaway_query',
           'qa_stock',
           'general_greet',
           'recommendation_events',
           'music_dislikeness',
           'iot_wemo_off',
           'cooking_recipe',
           'qa_currency',
           'transport_traffic',
           'general_quirky',
           'weather_query', 
           'audio_volume_up',
           'email_addcontact',
           'takeaway_order',
           'email_querycontact',
           'iot_hue_lightup',
           'recommendation_locations',
           'play_audiobook',
           'lists_createoradd',
           'news_query',
           'alarm_query',
           'iot_wemo_on',
           'general_joke',
           'qa_definition',
           'social_query',
           'music_settings',
           'audio_volume_other',
           'calendar_remove',
           'iot_hue_lightdim',
           'calendar_query',
           'email_sendemail',
           'iot_cleaning',
           'audio_volume_down',
           'play_radio',
           'cooking_query',
           'datetime_convert',
           'qa_maths',
           'iot_hue_lightoff',
           'iot_hue_lighton',
           'transport_query',
           'music_likeness',
           'email_query',
           'play_music',
           'audio_volume_mute',
           'social_post',
           'alarm_set',
           'qa_factoid',
           'calendar_set',
           'play_game',
           'alarm_remove',
           'lists_remove',
           'transport_taxi',
           'recommendation_movies',
           'iot_coffee',
           'music_query',
           'play_podcasts',
           'lists_query'
           ]

ID2LABEL = {i:label for i, label in enumerate(INTENTS)}
LABEL2ID = {label:i for i, label in enumerate(INTENTS)}
NLU_CONFIDENCE_THRESHOLD = 0.8
NLU_MODEL_NAME = "./massive_intent_recognition_model/checkpoint-8640"


#- load model on device-#
try:
    NLU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NLU on device : {NLU_DEVICE}")

    NLU_TOKENIZER = AutoTokenizer.from_pretrained(NLU_MODEL_NAME)

    NLU_MODEL = AutoModelForSequenceClassification.from_pretrained(
        NLU_MODEL_NAME,
        num_labels = len(INTENTS),
        id2label = ID2LABEL,
        label2id = LABEL2ID,
        ignore_mismatched_sizes = True
    ).to(NLU_DEVICE).eval()

    print (f"Model {NLU_MODEL} loaded on {NLU_DEVICE}")

except Exception as e:
    print (f"NLU WARNING: error loading GPU/Model: {e}")
    NLU_MODEL=None
#-----------------------#

SLOT_MODEL_PATH = "./massive_slot_filling_model/checkpoint-2160"
SLOT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.getcwd()

try:
    SLOT_TOKENIZER = AutoTokenizer.from_pretrained(SLOT_MODEL_PATH)
    
    SLOT_MODEL = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_PATH).to(SLOT_DEVICE)

    ID2TAG = SLOT_MODEL.config.id2label 
    
    SLOT_MODEL.eval()
    print(f"Slot filler model loaded successfully. Total tags: {len(ID2TAG)}")

except Exception as e:
    print(f"Error loading slot model: {e}")
    SLOT_MODEL = None



#-----Extract Slots-----#
def extract_slots(text):
    if SLOT_MODEL is None:
        return {}
    
    command_text = clean_command_text(text)

    inputs = SLOT_TOKENIZER(
        command_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    ).to(SLOT_DEVICE)

    with torch.no_grad():
        outputs = SLOT_MODEL(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()
    
    if predictions.ndim == 0:
        predictions = np.expand_dims(predictions, axis=0)
        
    predicted_tags = [ID2TAG[p] for p in predictions]
    word_tokens = SLOT_TOKENIZER.convert_ids_to_tokens(inputs['input_ids'].squeeze().cpu().numpy())

    slots = {}
    current_slot_name = None
    current_slot_tokens = []

    for token, tag in zip(word_tokens, predicted_tags):
        if token in ['[CLS]', '[SEP]'] or token.startswith('[PAD]'):
            continue
            
        tag_prefix = tag[0]
        tag_name = tag[2:]

        # 🌟 PATCH (La stessa logica per correggere l'errore B- sul sub-token)
        if current_slot_name and token.startswith('##') and tag_prefix == 'B':
            tag_prefix = 'I'

        # Logica B (Begin) e O (Outside)
        if tag_prefix == 'B' or tag == 'O':
            # Se stiamo chiudendo uno slot precedente, salvalo
            if current_slot_name and current_slot_tokens:
                # 🌟 Ricostruzione intelligente 🌟
                raw_text = ""
                for t in current_slot_tokens:
                    # Aggiungi uno spazio se il token corrente NON è un sub-token
                    if not t.startswith('##') and raw_text != "":
                        raw_text += " "
                    # Rimuovi ## e aggiungi il pezzo
                    raw_text += t.replace("##", "")
                    
                slots[current_slot_name.lower()] = raw_text.strip()
                
            current_slot_name = None
            current_slot_tokens = []
            
            # Se il tag corrente è B, iniziamo un nuovo slot
            if tag_prefix == 'B':
                current_slot_name = tag_name
                current_slot_tokens = [token]

        # Logica I (Intermediate)
        elif tag_prefix == 'I':
            if tag_name == current_slot_name:
                current_slot_tokens.append(token)
            else:
                # Caso I-errato: Chiude lo slot precedente e riazzera
                if current_slot_name and current_slot_tokens:
                    raw_text = ""
                    for t in current_slot_tokens:
                        if not t.startswith('##') and raw_text != "":
                            raw_text += " "
                        raw_text += t.replace("##", "")
                    
                    slots[current_slot_name.lower()] = raw_text.strip()
                    current_slot_name = None
                    current_slot_tokens = []

    # Salva l'ultimo slot
    if current_slot_name and current_slot_tokens:
        raw_text = ""
        for t in current_slot_tokens:
            if not t.startswith('##') and raw_text != "":
                raw_text += " "
            raw_text += t.replace("##", "")
        
        slots[current_slot_name.lower()] = raw_text.strip()

    return slots
#--------------------#


#---classify stt text---#
def classify_intent(text):

    text_command = clean_command_text(text) 

    if NLU_MODEL is None or not text_command:
        return "unrecognized", 0.0
    
    inputs = NLU_TOKENIZER(text_command, return_tensors="pt", truncation=True, padding=True).to(NLU_DEVICE) 

    with torch.no_grad():
        outputs = NLU_MODEL(**inputs)

    probabilities = outputs.logits.softmax(dim=-1).max().item()
    predicted_class_id = outputs.logits.argmax().item()
    
    intent = ID2LABEL.get(predicted_class_id, "unrecognized") 
    
    if probabilities < NLU_CONFIDENCE_THRESHOLD:
        intent = "unrecognized"
        
    return intent, probabilities
#-----------------------#









#-backend main endpoint-#
@app.route('/process_audio',methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify({'status':'error','message':'No Audio file found'}),400
    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({'status':'error','message':'Filename empty'}),400
    
    fd,temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    try:
        audio_file.save(temp_path)
        print(f"File received and saved: {temp_path}")

        command_text = transcribe_audio(temp_path)
        print(f"Recognized text: {command_text}")

        if "ERROR" in command_text or not command_text:
            tts_response = 'An error has occurred in transcription'
            status_code = 'error'
        else:
            intent,confidence = classify_intent(command_text)
            print(f"Intent:{intent}, confidence:{confidence:.2f}")
            status_code = 'ok'
            tts_response = f"Command: {command_text}"
            if intent != 'unrecognized':
                slots = extract_slots(command_text)
                print(slots)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return jsonify({
        "status":status_code,
        "command_text":command_text,
        "tts_response":tts_response
    }),200
#-----------------------#



@app.route('/test',methods=['GET'])
def test():
    commands = [
        "play Bohemian Rhapsody",
        "Let me hear Rock music",
        "What is the weather is Paris?",
        "Put an alarm for tomorrow ad 9 am",
        "Tell me a good comic movie"
    ]

    for command in commands:
        intent,confidence = classify_intent(command)
        print(command)
        print(f"Intent:{intent}, confidence:{confidence:.2f}")
        status_code = 'ok'
        tts_response = f"Command: {command}"
        if intent != 'unrecognized':
            slots = extract_slots(command)
            print(slots)

    return jsonify({
        "status":'ok'
    }),200