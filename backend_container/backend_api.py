from flask import Flask,request,jsonify
import torch
import os
import wave
import json
import tempfile
from vosk import Model,KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

app = Flask(__name__)
MODEL_PATH = "vosk-model-it"

print(f"CUDA disponibile in PyTorch: {torch.cuda.is_available()}")
print(f"GPU Trovata: {torch.cuda.get_device_name(0)}")

#--- load stt model ----#
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    vosk_model = Model(MODEL_PATH)
    print("Model loeaded successfully")

except Exception as e:
    print (f"Critical Error: {e}")
    vosk_model=None
#-----------------------#


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



INTENTS = [
    'riproduci_musica',
    'chat_generale',
    'goodbye',
    'unrecognized'
]
ID2LABEL = {i:label for i, label in enumerate(INTENTS)}
LABEL2ID = {label:i for i, label in enumerate(INTENTS)}
NLU_CONFIDENCE_THRESHOLD = 0.8
NLU_MODEL_NAME = "./nlu_model_finetuned"


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

SLOT_MODEL_PATH = "./slot_model_finetuned"
SLOT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2TAG = {0:'0',1:'B-GENERE',2:'I-GENERE',3:'B-ARTISTA',4:'I-ARTISTA',5:'B-ANNO',6:'I-ANNO'}

try:
    SLOT_TOKENIZER = AutoTokenizer.from_pretrained(SLOT_MODEL_PATH)
    SLOT_MODEL = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_PATH).to(SLOT_DEVICE)
    SLOT_MODEL.eval()
    print("Slot filler model loaded successfully")
except Exception as e:
    print(f"Error loading slot model: {e}")
    SLOT_MODEL= None

def extract_slots(command_text):
    if SLOT_MODEL is None:
        return {}

    inputs = SLOT_TOKENIZER(
        command_text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    ).to(SLOT_DEVICE)

    with torch.no_grad():
        outputs = SLOT_MODEL(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()
    
    predicted_tags = [ID2TAG[p] for p in predictions]
    
    word_tokens = SLOT_TOKENIZER.convert_ids_to_tokens(inputs['input_ids'].squeeze().cpu().numpy())
    
    slots = {}
    current_slot_name = None
    current_slot_value = []
    
    for token, tag in zip(word_tokens, predicted_tags):
        if token in [SLOT_TOKENIZER.cls_token, SLOT_TOKENIZER.sep_token] or tag == 'O' or token.startswith('[PAD]'):
            if current_slot_name and current_slot_value:
                slots[current_slot_name.lower()] = " ".join(current_slot_value).replace(" ##", "")
            current_slot_name = None
            current_slot_value = []
            continue

        if token.startswith('##'):
            word_part = token[2:]
        else:
            word_part = token

        tag_prefix = tag[0]
        tag_name = tag[2:]

        if tag_prefix == 'B':
            if current_slot_name and current_slot_value:
                slots[current_slot_name.lower()] = " ".join(current_slot_value).replace(" ##", "")
            
            current_slot_name = tag_name
            current_slot_value = [word_part]
            
        elif tag_prefix == 'I':
            if tag_name == current_slot_name:
                current_slot_value.append(word_part)
            else:
                pass

    if current_slot_name and current_slot_value:
        slots[current_slot_name.lower()] = " ".join(current_slot_value).replace(" ##", "")

    return slots


#---classify stt text---#
def classify_intent(text_command):

    if NLU_MODEL is None or not text_command:
        return "unrecognized",0.0
    
    inputs = NLU_TOKENIZER(text_command, return_tensors="pt",truncation=True,padding=True).to(NLU_DEVICE)

    with torch.no_grad():
        outputs = NLU_MODEL(**inputs)

    probabilities = outputs.logits.softmax(dim=-1).max().item()
    predicted_class_id = outputs.logits.argmax().item()
    
    intent = ID2LABEL.get(predicted_class_id, "unrecognized")   
    
    if probabilities < NLU_CONFIDENCE_THRESHOLD:
        intent = "unrecognized"
    return intent,probabilities
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
            if intent == 'riproduci_musica':
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


@app.route('/',methods=['GET'])
def greet():
    return jsonify({'status':'ok','message':'all good'}),200


