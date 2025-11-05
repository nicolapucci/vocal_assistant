import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    IntervalStrategy,
    )
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---Load Data
BASE_DIR = os.getcwd()
data_files = {
    "train":os.path.join(BASE_DIR,'train','0000.parquet'),
    "test":os.path.join(BASE_DIR,'test','0000.parquet'),
    "validation":os.path.join(BASE_DIR,'validation','0000.parquet'),
}
dataset_en_US = load_dataset("parquet",data_files=data_files)
#----------


#---Intents
INTENTS = ['datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock',
           'general_greet', 'recommendation_events', 'music_dislikeness', 'iot_wemo_off', 'cooking_recipe',
           'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query', 'audio_volume_up',
           'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup',
           'recommendation_locations', 'play_audiobook', 'lists_createoradd', 'news_query',
           'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query',
           'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim',
           'calendar_query', 'email_sendemail', 'iot_cleaning', 'audio_volume_down',
           'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff',
           'iot_hue_lighton', 'transport_query', 'music_likeness', 'email_query', 'play_music',
           'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid', 'calendar_set',
           'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies',
           'iot_coffee', 'music_query', 'play_podcasts', 'lists_query']

ID2LABEL = {i: label for i, label in enumerate(INTENTS)} 
LABEL2ID = {label: i for i, label in enumerate(INTENTS)}
NUM_INTENTS = len(INTENTS)

all_intent_ids = []
for split in dataset_en_US.keys():
    all_intent_ids.extend(dataset_en_US[split]["intent"])

UNIQUE_INTENT_IDS = sorted(list(set(all_intent_ids))) 

ORIG_TO_CONTIGUOUS = {orig_id: i for i, orig_id in enumerate(UNIQUE_INTENT_IDS)}
#----------


#----------
def tokenize_function(examples):
    examples["label"] = [ORIG_TO_CONTIGUOUS[orig_id] for orig_id in examples["intent"]]

    tokenized_inputs = tokenizer(examples['utt'],padding="max_length",truncation=True)

    return tokenized_inputs
#----------

#----------
MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(UNIQUE_INTENT_IDS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(DEVICE)

tokenized_dataset = dataset_en_US.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset_en_US["train"].column_names
)
#----------

#----------
def compute_metrics(p):
    """
    Calcola l'accuratezza, la precisione, la recall e l'F1-score per la classificazione di sequenze.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0,
    )

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
#----------

OUTPUT_DIR = "./massive_intent_recognition_model"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,

    logging_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs_intent', 
    logging_steps=100,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting Intent Recognition Fine-Tuning...")

try:
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Modello Intent Recognition trained and saved: {OUTPUT_DIR}")

except Exception as e:
    print(f"PORCODIO")