from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy,
)
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
import os
from pathlib import Path

# Definiamo la root del progetto in modo robusto, assumendo che questo file sia
# in /root/trainer/trainers/intent_recognition_trainer/trainer.py
# Risaliamo di 4 livelli: [trainer.py] -> [intent_recognition_trainer] -> [trainers] -> [trainer] -> [root]
# AGGIORNATO: Assumendo che il file sia in /root/intent_recognition_trainer/trainer.py (risalita di 2)
# Usiamo il PROJECT_ROOT che il training_handler ha definito.
# Per sicurezza, ricreiamo la logica di percorso assoluto per MODEL_STORAGE
# Nota: La cartella 'model-artifacts' si trova due livelli sopra questo file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- La funzione get_default_model_name e il compute_metrics sono corretti e rimangono invariati. ---

def get_default_model_name(MODEL_STORAGE):
    MODEL_PREFIX = "XLM-roBERTa-v"
    count = 0
    # Usiamo glob per iterare su Path, è più pulito e gestisce meglio gli oggetti Path
    for item in MODEL_STORAGE.iterdir():
        if item.is_dir() and item.name.startswith(MODEL_PREFIX):
            count += 1
    name = f"{MODEL_PREFIX}{count}"
    return name

def compute_metrics(p):
    predictions,labels = p
    predictions = np.argmax(predictions,axis=1)

    accuracy = accuracy_score(labels, predictions)

    # Nota: average='weighted' è corretto per l'Intent Recognition con classi sbilanciate
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall
    }

def train_intent_recognition(training_settings:dict):

    # Percorso ASSOLUTO corretto per lo storage dei modelli
    MODEL_STORAGE = PROJECT_ROOT / 'model-artifacts' / 'intent-recognition'
    
    # -------------------- VALIDAZIONE E IMPOSTAZIONI --------------------
    MODEL_PATH = training_settings['model_path']
    if MODEL_PATH is None:
        raise ValueError(f"Model Path is missing")
    
    name = training_settings['name'] if training_settings['name'] else get_default_model_name(MODEL_STORAGE)

    metric_for_best_model = training_settings.get('metric_for_best_model', 'eval_loss')
    greater_is_better = False if metric_for_best_model=='eval_loss' else True

    weight_decay = training_settings.get('weight_decay', 0.01)

    early_stopping_patience = training_settings.get('early_stopping_patience', 5)
    early_stopping_threshold = training_settings.get('early_stopping_threshold', 0.001)
        
    dataset = training_settings['dataset']
    if dataset is None:
        raise ValueError(f"Dataset is missing")
    
    # !!! CORREZIONE CRITICA: Aumento del Learning Rate !!!
    # Default corretto per il fine-tuning di XLM-RoBERTa: 1e-5 (0.00001)
    learning_rate = training_settings.get('learning_rate', 1e-5) 
    
    num_epochs = training_settings.get('num_epochs', 3)
    intents = training_settings['intents'] # Usiamo il dizionario completo Intents
    
    # -------------------- PREPARAZIONE DATI E LABELS --------------------
    all_intent_ids = []
    for split in dataset.keys():
        all_intent_ids.extend(dataset[split]['intent'])
    
    # Ottiene gli ID originali unici e li ordina
    intents_ids = sorted(list(set(all_intent_ids))) 
    # Mappa l'ID originale all'indice contiguo (0, 1, 2, ...)
    orig_to_contiguous = {orig_id:i for i,orig_id in enumerate(intents_ids)} 

    # Usa il dizionario intents completo per ottenere i nomi degli intenti in ordine contiguo
    contiguous_intent_names = [intents[orig_id] for orig_id in intents_ids]

    id2label = {i: label for i,label in enumerate(contiguous_intent_names)}
    label2id = {label: i for i,label in enumerate(contiguous_intent_names)}

    def tokenize_function(examples):
        # Mappa gli intent ID originali all'ID contiguo per il modello
        examples['label']= [orig_to_contiguous[orig_id] for orig_id in examples['intent']]
        tokenized_inputs = tokenizer(examples['utt'],padding='max_length',truncation=True)
        return tokenized_inputs

    # -------------------- CARICAMENTO MODELLO E TOKENIZER --------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Se MODEL_PATH è un oggetto Path, viene gestito correttamente da from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(intents_ids),
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        # Rimuove le colonne originali non necessarie al Trainer
        remove_columns=dataset['train'].column_names 
    )

    # -------------------- ARGOMENTI DI TRAINING E SETUP --------------------
    output_dir = MODEL_STORAGE / name
    
    training_args = TrainingArguments(
        output_dir = str(output_dir), # Converti Path in str per TrainingArguments
        num_train_epochs=num_epochs,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),

        logging_strategy=IntervalStrategy.EPOCH,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,

        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_dir=str(PROJECT_ROOT / 'training-logs' / name), # Usa Pathlib
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
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )]
    )

    # -------------------- ESECUZIONE --------------------
    try:
        results = trainer.train()
        # Salva la best model (che è già stata caricata in memoria grazie a load_best_model_at_end=True)
        final_save_path = output_dir / 'production_candidate'
        trainer.save_model(str(final_save_path)) 
        
        return f"Model successfully trained. Saved in {final_save_path}"
        
    except Exception as e:
        # Se c'è un errore nel training, solleva l'eccezione, non stampare solo
        raise RuntimeError(f"Error during model training: {e}")