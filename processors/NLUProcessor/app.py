import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from flask import (
    Flask,
    request,
    jsonify
)
from pathlib import Path

cwd = Path(__name__).parent
app = Flask(__name__)
PORT = 5002

intent_mapper = [
        'datetime_query','iot_hue_lightchange','transport_ticket','takeaway_query','qa_stock','general_greet',
        'recommendation_events','music_dislikeness','iot_wemo_off','cooking_recipe','qa_currency','transport_traffic',
        'general_quirky','weather_query','audio_volume_up','email_addcontact','takeaway_order','email_querycontact',
        'iot_hue_lightup','recommendation_locations','play_audiobook','lists_createoradd','news_query','alarm_query',
        'iot_wemo_on','general_joke','qa_definition','social_query','music_settings','audio_volume_other','calendar_remove',
        'iot_hue_lightdim','calendar_query','email_sendemail','iot_cleaning', 'audio_volume_down','play_radio','cooking_query',
        'datetime_convert','qa_maths','iot_hue_lightoff','iot_hue_lighton','transport_query','music_likeness','email_query',
        'play_music','audio_volume_mute','social_post','alarm_set','qa_factoid','calendar_set','play_game','alarm_remove',
        'lists_remove','transport_taxi','recommendation_movies','iot_coffee','music_query','play_podcasts','lists_query'
           ]

def load_slot_labels(slot_dir:Path):  
    try:
        model = AutoModelForTokenClassification.from_pretrained(slot_dir)
            
        id2label = model.config.id2label
            
        slot_labels = {int(k): str(v) for k, v in id2label.items()}
            
        return slot_labels
            
    except Exception as e:
        print(f"Errore durante il caricamento delle etichette slot: {e}")
        return {}
    

class NLUProcessor:

    def __init__(self):

        intent_dir = cwd.parent / 'models' / 'intent-recognition'
        slot_dir = cwd.parent / 'models' / 'slot-filling'

        print(f"Loading NLU models")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.slot_tokenizer = AutoTokenizer.from_pretrained(slot_dir)
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_dir)

        self.slot_model = AutoModelForTokenClassification.from_pretrained(slot_dir)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_dir)

        self.slot_model.to(self.device)
        self.intent_model.to(self.device)

        self.slot_labels = load_slot_labels(slot_dir)





    def _extract_slots(self, tokens, slot_predictions):
        extracted_slots = []
        current_slot_key = None
        current_slot_value = []
        
        for token, pred_index in zip(tokens, slot_predictions):
            slot_tag = self.slot_labels.get(pred_index, "O")

            if token in ['[CLS]', '[SEP]']:
                continue
            
            clean_token = token.replace('##', '')

            if slot_tag.startswith('B-') or slot_tag == 'O':
                if current_slot_key and current_slot_value:
                    value = "".join(current_slot_value).strip()
                    extracted_slots.append({current_slot_key: value})
                
                current_slot_key = None
                current_slot_value = []

            if slot_tag.startswith('B-'):
                current_slot_key = slot_tag[2:]
                current_slot_value.append(clean_token)
            elif slot_tag.startswith('I-') and current_slot_key:
                if slot_tag[2:] == current_slot_key:
                    current_slot_value.append(clean_token)

        if current_slot_key and current_slot_value:
            value = "".join(current_slot_value).strip()
            extracted_slots.append({current_slot_key: value})
            
        return extracted_slots
    

    def process_text(self,text):

        inputs = self.intent_tokenizer(text,return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        intent_outputs = self.intent_model(**inputs)

        intent_prediction = torch.argmax(intent_outputs.logits, dim=1).item()
        intent = intent_mapper[intent_prediction] if intent_mapper[intent_prediction] else 'unknown'

        if intent != "unknown":
            
            slot_inputs = self.slot_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            slot_inputs = {k: v.to(self.device) for k, v in slot_inputs.items()}
            
            slot_outputs = self.slot_model(**slot_inputs)
            
            slot_predictions = torch.argmax(slot_outputs.logits, dim=2)[0].tolist()
            
            tokens = self.slot_tokenizer.convert_ids_to_tokens(slot_inputs['input_ids'][0])
            
            slots = self._extract_slots(tokens, slot_predictions)
        else:
            slots = []
            
        return intent, slots
    
nlu_processor = NLUProcessor()

@app.route('/',methods=['POST'])
def extract_intent_and_slots():
    if not request.is_json or 'text' not in request.json:
        print('errore')
        return jsonify({'ok':False,'content':"C'è stato un errore"}),400
    text =request.json['text']
    try:
        if text:
            print(f"input: {text}")
            intent,slots = nlu_processor.process_text(text)
        return jsonify({'ok':True,'content':{'slots':slots,'intent':intent}})
    except Exception:
        return jsonify({'ok':False,'content':"C'è stato un errore"})
    
if __name__ == '__main__':
    print(f"Avvio del server di configurazione su http://localhost:{PORT}")
    app.run(port=PORT, debug=True)