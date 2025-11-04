import pandas as pd
from datasets import Dataset
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

INTENTS = [
    'riproduci_musica',
    'chat_generale',
    'goodbye',
    'unrecognized'
]

ID2LABEL = {i: label for i, label in enumerate(INTENTS)}
LABEL2ID = {label: i for i, label in enumerate(INTENTS)}

#to be updated with './nlu_model_finetuned' later
MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased" 
DATA_FILE = "nlu_data.csv"
OUTPUT_DIR = "./nlu_model_finetuned" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Statring Training. Target device: {DEVICE}")


def tokenize_function(examples):
    examples["label"] = LABEL2ID[examples["label"]]
    return tokenizer(examples["text"],padding="max_length",truncation = True)

#compute accuracy and precision
def compute_metrics(p):
    predictions,labels = p
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        labels=np.unique(predictions)
    )

    return {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall
    }


#----Data tokenization and loading----#
try:
    df = pd.read_csv(DATA_FILE)
    
    # PULIZIA DATI AGGIUNTA QUI:
    # 1. Rimuovi righe con valori mancanti (NaN) in qualsiasi colonna.
    df.dropna(inplace=True)
    
    # 2. Converti testo e label in stringhe e rimuovi eventuali spazi bianchi superflui.
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    
    # 3. Filtra solo le etichette valide presenti nel tuo array INTENTS
    valid_labels = set(INTENTS)
    df = df[df['label'].isin(valid_labels)]
    
    print(f"Dataset caricato: {len(df)} righe valide.")
    dataset = Dataset.from_pandas(df)

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

#initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(INTENTS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(DEVICE)#move the model to the gpu (if it doens't find the gpu then it moves it to the cpu)

#tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=False)

#split dataset for training(80%) and test(20%)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]
#-------------------------------------#


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=6,                 # number of training epochs
    per_device_train_batch_size=8,      # low to not overload gpu
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,

    fp16=True,

    logging_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True,        #load model with best performance
    metric_for_best_model="accuracy",   #use accuracy to choose best model
    logging_dir='./logs',
    logging_steps=100,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning. extimated time:30-60 minutes")

trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"Trained NLU model saved in: {OUTPUT_DIR}")