from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy,
)

import torch
import numpy as np
import os
import re
import evaluate
from pathlib import Path

current_file_dir = Path(__file__).parent

metric = evaluate.load('seqeval')

def get_default_model_name(MODEL_STORAGE):
    MODEL_PREFIX = "XLM-roBERTa-v"
    count = 0
    for item in os.listdir(MODEL_STORAGE):
        full_path = os.path.join(MODEL_STORAGE,item)
        if os.path.isdir(full_path) and item.startswith(MODEL_PREFIX):
            count += 1
    name = f"{MODEL_PREFIX}{count}"
    return name

def train_slot_filling(training_settings:dict):

    MODEL_STORAGE = os.path.join(current_file_dir,'../../model-artifacts/slot-filling')

    MODEL_PATH = training_settings['model_path']
    if MODEL_PATH is None:
        raise ValueError(f"Model Name is missing")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    name = training_settings['name'] if training_settings['name'] else get_default_model_name(MODEL_STORAGE)

    metric_for_best_model = training_settings['metric_for_best_model'] if training_settings['metric_for_best_model'] else 'eval_loss'
    greater_is_better = False if metric_for_best_model=='eval_loss' else True

    weight_decay = training_settings['weight_decay'] if training_settings['weight_decay'] else 0.01

    warmup_steps = training_settings['warmup_steps'] if training_settings['warmup_steps'] else 300

    early_stopping_patience = training_settings['early_stopping_patience'] if training_settings['early_stopping_patience'] else 5
    early_stopping_threshold = training_settings['early_stopping_threshold'] if training_settings['early_stopping_threshold'] else 0.001
        
    dataset = training_settings['dataset']
    if dataset is None:
        raise ValueError(f"Dataset is missing")
    
    learning_rate = training_settings['learning_rate'] if training_settings['learning_rate'] else 1e-7
    num_epochs = training_settings['num_epochs'] if training_settings['num_epochs'] else 3

#==============
    all_annot_utts = []
    for split in dataset.keys():
        all_annot_utts.extend(dataset[split]['annot_utt'])
    
    slot_pattern = re.compile(r'\[\s*(.+?)\s*:\s*.+?\s*\]')

    raw_slot_names_nested = [slot_pattern.findall(text) for text in all_annot_utts]

    raw_slot_names = sorted(list(set(
        item.strip() for sublist in raw_slot_names_nested for item in sublist
    )))

    label_list = ['O']
    for slot in raw_slot_names:
        label_list.append(f"B-{slot}")
        label_list.append(f"I-{slot}")

    id2label = {i:label for i, label in enumerate(label_list)}
    label2id = {label: i for i,label in enumerate(label_list)}
    num_labels = len(label_list)
#==============

#==============
    ANNOT_COLUMN_NAME = "annot_utt" 
    NEW_LABEL_COLUMN = "labels"
    def parse_and_align_labels(examples):
        all_aligned_labels =  []
        all_tokenized_inputs = []

        for annot_text in examples[ANNOT_COLUMN_NAME]:
            segments = re.findall(r'(\[\s*(.+?)\s*:\s*(.+?)\s*\])|([^\[\]]+)', annot_text)

            words = []
            bio_labels = []

            for segment in segments:
                slot_info = segment[1].strip()
                slot_value = segment[2].strip()
                raw_text = segment[3].strip()

                if slot_value and slot_info:
                    slot_words = slot_value.split()
                    if not slot_words:
                        continue

                    words.append(slot_words[0])
                    bio_labels.append(label2id.get(f"B-{slot_info}",label2id['O']))

                    for word in slot_words[1:]:
                        words.append(word)
                        bio_labels.append(label2id.get(f"I-{slot_info}",label2id['O']))
                
                elif raw_text:
                    raw_words = raw_text.split()
                    words.extend(raw_words)
                    bio_labels.extend([label2id['O']]*len(raw_words))
            
            tokenized_output = tokenizer(words, truncation=True, is_split_into_words = True)
            word_ids = tokenized_output.word_ids(batch_index=0)

            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx == previous_word_idx:
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(bio_labels[word_idx])
                previous_word_idx = word_idx
            
            all_aligned_labels.append(aligned_labels)
            all_tokenized_inputs.append(tokenized_output)
        
        final_inputs = {}
        for key in all_tokenized_inputs[0].keys():
            final_inputs[key] =  [item[key] for item in all_tokenized_inputs]

        final_inputs[NEW_LABEL_COLUMN] = all_aligned_labels
        return final_inputs
    
    tokenized_datasets = dataset.map(
        parse_and_align_labels,
        batched=True
    )
#==============

#==============


    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes = True
    ).to(DEVICE)
#==============

#==============
    def compute_metrics(p):
        predictions,labels = p
        predictions = np.argmax(predictions,axis=2)

        true_predictions = [
            [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
            for prediction, label in zip(predictions,labels)
        ]
        true_labels = [
            [label_list[l] for (p,l) in zip(prediction,label) if l!=-100]
            for prediction, label in zip(predictions,labels)
        ]

        results = metric.compute(
            predictions = true_predictions,
            references=true_labels,
            zero_division=0
        )

        return { 
            "f1": results["overall_f1"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "accuracy": results["overall_accuracy"],
        }
#==============

#==============
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_STORAGE,name),
        
        num_train_epochs=num_epochs,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        
        logging_dir=os.path.join(current_file_dir,'training-logs',name),
        logging_steps=100,

        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,

        load_best_model_at_end=True,

        learning_rate=learning_rate,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,

        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_patience=early_stopping_patience
        )]
    )

    try:
        print(len(tokenized_datasets['train']))
        trainer.train()
        trainer.save_model(os.path.join(MODEL_STORAGE,name,'production_candidate'))
    except Exception as e:
        print(f"Something went wrong. Check the settings and try again")
#==============

    return f"Model successfully trained. Saved in{os.path.join(MODEL_STORAGE,name,'production_candidate')}"
