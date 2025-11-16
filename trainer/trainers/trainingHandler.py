from scripts import data_loader
from intent_recognition_trainer.trainer import train_intent_recognition as intentTrainer
from slot_filling_trainer.trainer import train_slot_filling as slotTrainer

import argparse
import os
from pathlib import Path
import sys

"""
training_setting = { <-dict
                    model_path : path to the model if in local or name of the model if pulled from huggingface, MANDATORY
                    name: name of the model, can autogenerete one if not provided
                    metric_for_best_model: eval_loss/eval_f1/accuracy, default is eval_loss
                    weight_decay: model weight decay, default is 0.01
                    early_stopping_patience: how many times trigger the threshold before stop training, default is 5
                    early_stopping_threshold: threshold that triggers patiente during training, default is 0.001
                    dataset: dataset, MANDATORY
                    intents: intents, MANDATORY ONLY for intent recognition
                    learning_rate: learnig rate, default is 1e-7
                    num_epochs: training cicles, default is 3
                    }
"""
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTENT_MODEL_STORAGE = PROJECT_ROOT / 'model-artifacts' / 'intent-recognition'
SLOT_MODEL_STORAGE = PROJECT_ROOT / 'model-artifacts' / 'slot-filling'

parser = argparse.ArgumentParser(description="Script for NLU model training (Intent Rec. and Slot Filling)")
parser.add_argument(
    '--model_path',
    type=str,
    default=None,#if no --model_path is provided then use base XLM-roBERTa model
    help='Path o name of Hugging Face model to use'
)
parser.add_argument(
    '--name',
    type=str,
    default=None,
    help='Name of the resulting model'
)
parser.add_argument(
    '--metric',
    type=str,
    default='eval_loss',
    help='metric used to choose the best model'
)
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.01,
    help='decay of the model weights'
)
parser.add_argument(
    '--es_patience',
    type=int,
    default=5,
    help='How many times the model is allowed not to reach the threshold before stopping'
)
parser.add_argument(
    '--es_threshold',
    type=float,
    default=0.001,
    help='If the model does not improve this much it will trigger the patience counter'
)
parser.add_argument(
    '--custom_csv',
    action='store_true',
    help='include the custom csv for the dataset'
)
parser.add_argument(
    '--no_massive',
    action='store_true',
    help='include the AmazonScience/massive dataset'

)
parser.add_argument(
    '--massive_RI',
    type=int,
    default=None,
    help='How many of massive dataset entry with relevant intents to use'
)
parser.add_argument(
    '--massive_NRI',
    type=int,
    default=None,
    help='How many of massive dataset entry with non relevant intents to use'
)
parser.add_argument(
    '--lr',
    type=float,
    default=2e-5,
    help='Learning Rate'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='Number of training epochs'
)
parser.add_argument(
    '--warmup',
    type=int,
    default=300,
    help='Warmup for the model b4 the training'
)
parser.add_argument(
    '--intent_trainer',
    action='store_true',
    help='If present, use the Intent Recognition trainer, if not uses the Slot Filling trainer'
)

args = parser.parse_args()
if not args.custom_csv and args.no_massive:
    raise ValueError(f"Train Dataset cannot be empty")

#define trainer, storage of the models
if args.intent_trainer:
    trainer_func = intentTrainer
    model_storage = INTENT_MODEL_STORAGE
    task_name = "Intent Recognition"
    intents_required = True
    warmup_required = False
else:
    trainer_func = slotTrainer
    model_storage = SLOT_MODEL_STORAGE
    task_name = "Slot Filling"
    intents_required = False
    warmup_required = True

#default value is xlm-roberta-base, if path(name of the folder of the model) is specified and exists then use the custom model
model_path = None
if args.model_path is None:
    model_path = 'xlm-roberta-base'
else:
    model_path = model_storage / args.model_path / 'production_candidate'
    if not model_path.exists():
        print(f"Could not find model")
        sys.exit(1)
    model_path = str(model_path)

name = args.name
if name is not None:
    path = model_storage / name
    if path.exists():
        print(f"Invalid name, there is already a model with that name")
        sys.exit(1)

#by default load all the dataset, for now isn't not possible to create subsets of massive dataset
try:
    dataset = data_loader.create_custom_dataset(
        use_custom=True if args.custom_csv else False,
        use_massive=True if args.no_massive is False else False,
    )#i'll handle relevant intents later
except Exception as e:
    print(f"Critical Error loading data: {e}")
    sys.exit(1)

training_settings = {
    "model_path":model_path,
    "name":args.name,
    "metric_for_best_model":args.metric,
    "weight_decay":args.weight_decay,
    "early_stopping_patience":args.es_patience,
    "early_stopping_threshold":args.es_threshold,
    "dataset":dataset,
    "intents":data_loader.get_intents_mapping() if intents_required else None,#intent is only needed for intent recognition
    "warmup_steps":args.warmup if warmup_required else None,
    "learning_rate":args.lr,
    "num_epochs":args.epochs,
}

try:
    results = trainer_func(training_settings)
except Exception as e:
    print(f"Error traninig the model: {e}")
    sys.exit(1)

print(results)