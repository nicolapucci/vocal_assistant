import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    IntervalStrategy
)
import torch
import numpy as np
from seqeval.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score)
from sklearn.model_selection import train_test_split

ENTITY_TAGS = [
    'O','B-GENERE','I-GENERE','B-ARTISTA','I-ARTISTA','B-ANNI','I-ANNI'
]

ID2TAG = {i:tag for i,tag in enumerate(ENTITY_TAGS)}
TAG2ID = {tag:i for i,tag in enumerate(ENTITY_TAGS)}

MODEL_NAME = 'dbmdz/bert-base-italian-xxl-cased'

OUTPUT_DIR = "./slot_model_finetuned"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Statring Slot Filler Training. Target device: {DEVICE}")


#---align tags and tokens---#
def tokenize_and_align_labels(examples):
    global tokenizer
    tokenized_inputs = tokenizer(examples["word"], truncation=True, is_split_into_words=True)
    labels = []
    
    for i, label in enumerate(examples["tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(TAG2ID[label[word_idx]])
            elif word_idx == previous_word_idx:
                label_ids.append(-100)
                
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
#---------------------------#


#------compute metrics------#
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        
        true_label = [ID2TAG[l] for l in label if l != -100]
        
        true_prediction = [ID2TAG[p] for p, l in zip(prediction, label) if l != -100]
        
        true_labels.append(true_label)
        true_predictions.append(true_prediction)

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions,zero_division=0),
        "precision": precision_score(true_labels, true_predictions,zero_division=0),
        "recall": recall_score(true_labels, true_predictions,zero_division=0),
    }
#---------------------------#


#-------Data loading--------#
def load_conll_data(file_path):
    sentences, tags = [], []
    current_sentence, current_tags = [], []

    with open(file_path, 'r', encoding='utf-8') as f:

        try:
            next(f) 
        except StopIteration:
            pass
        for line in f:
            line = line.strip()
            

            if not line: 
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                current_sentence, current_tags = [], []
            else:
                parts = line.split('\t')
                
                if len(parts) == 2:
                    word, tag = parts
                    current_sentence.append(word.strip())
                    current_tags.append(tag.strip())

    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    
    return sentences, tags


try:
    SLOT_DATA_FILE = "slot_data.csv"
    
    sentences, tags = load_conll_data(SLOT_DATA_FILE)
    
    if not sentences:
        raise ValueError("Nessuna frase valida trovata nel dataset. Controllare il formato (tabulatore) e le righe vuote.")

    data = {'word': sentences, 'tag': tags}
    print(f"Dataset Slot Filler caricato: {len(data['word'])} frasi valide.")
    
    if len(data['word']) < 2:
         raise ValueError(f"Trovata solo {len(data['word'])} frase/i. Aggiungere più dati per lo split train/test (minimo 2).")

    dataset = Dataset.from_dict(data)

except Exception as e:
    print(f"Errore critico durante il caricamento dei dati: {e}")
    exit()
#---------------------------#


#-------Initialization------#
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(ENTITY_TAGS),
    id2label=ID2TAG,
    label2id=TAG2ID,
).to(DEVICE)

tokenized_dataset  = dataset.map(tokenize_and_align_labels,batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["word", "tag"])

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2,seed=42)

train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']
#---------------------------#

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    logging_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./slot_logs',
    report_to=None,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("Starting Slot Filler fine-tuning.")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"Trained Slot Filler model saved in: {OUTPUT_DIR}")