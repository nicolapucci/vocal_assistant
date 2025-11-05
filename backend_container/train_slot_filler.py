from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments,
                          Trainer,
                          IntervalStrategy
                          )
from datasets import load_dataset
import torch
import numpy as np
import evaluate
import os
import re
import itertools

BASE_DIR = os.getcwd()

metric = evaluate.load("seqeval")

MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Statring Training. Target device: {DEVICE}")

#---DATASET
data_files = {
    "train":os.path.join(BASE_DIR,'train','0000.parquet'),
    "test":os.path.join(BASE_DIR,'test','0000.parquet'),
    "validation":os.path.join(BASE_DIR,'validation','0000.parquet'),
}
dataset_en_US = load_dataset("parquet",data_files=data_files)
#--------


#---LABELS
all_annot_utts = []
for split in dataset_en_US.keys():
    all_annot_utts.extend(dataset_en_US[split]["annot_utt"])

slot_pattern = re.compile(r'\[\s*(.+?)\s*:\s*.+?\s*\]') 

raw_slot_names_nested = [slot_pattern.findall(text) for text in all_annot_utts]

RAW_SLOT_NAMES = sorted(list(set(
    item.strip() for sublist in raw_slot_names_nested for item in sublist
)))

label_list = ['O'] 
for slot in RAW_SLOT_NAMES:
    label_list.append(f'B-{slot}')
    label_list.append(f'I-{slot}')

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
NUM_LABELS = len(label_list)
#----------


#---Tokenizer
ANNOT_COLUMN_NAME = "annot_utt" 
NEW_LABEL_COLUMN = "labels" 

def parse_and_align_labels(examples):
    all_aligned_labels = []
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
                if not slot_words: continue

                words.append(slot_words[0])
                bio_labels.append(label2id.get(f'B-{slot_info}', label2id['O'])) 
                
                for word in slot_words[1:]:
                    words.append(word)
                    bio_labels.append(label2id.get(f'I-{slot_info}', label2id['O']))
            
            elif raw_text:
                raw_words = raw_text.split()
                words.extend(raw_words)
                bio_labels.extend([label2id['O']] * len(raw_words))

        tokenized_output = tokenizer(words, truncation=True, is_split_into_words=True)
        word_ids = tokenized_output.word_ids(batch_index=0)
        
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx == previous_word_idx:
                aligned_labels.append(bio_labels[word_idx])
            else:
                aligned_labels.append(bio_labels[word_idx])
            previous_word_idx = word_idx

        all_aligned_labels.append(aligned_labels)
        all_tokenized_inputs.append(tokenized_output)

    final_inputs = {}
    for key in all_tokenized_inputs[0].keys():
        final_inputs[key] = [item[key] for item in all_tokenized_inputs]
        
    final_inputs[NEW_LABEL_COLUMN] = all_aligned_labels
    return final_inputs

tokenized_datasets = dataset_en_US.map(
    parse_and_align_labels, 
    batched=True,
    remove_columns=dataset_en_US["train"].column_names 
)
#----------

#--Load Model and Collator
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS, 
    id2label=id2label, 
    label2id=label2id
).to(DEVICE)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
#----------


#----------
def compute_metrics(p):
    predictions, labels = p

    predictions = np.argmax(predictions,axis=2)

    true_predictions = [
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]

    results = metric.compute(predictions=true_predictions,references=true_labels)

    return { 
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "accuracy": results["overall_accuracy"],
    }
#-------

#---Training
training_args = TrainingArguments(
    output_dir="./massive_slot_filling_model",
    num_train_epochs=3,                     
    per_device_train_batch_size=16,            
    per_device_eval_batch_size=16,
    warmup_steps=500,                         
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy=IntervalStrategy.EPOCH,           
    save_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
#-------
