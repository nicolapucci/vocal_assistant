from datasets import (load_dataset,
                      DatasetDict,
                      concatenate_datasets,
                      ClassLabel
                      )
import os
from pathlib import Path

current_file_dir = Path(__file__).parent
PARQUET_BASE_DIR = os.path.join(current_file_dir,'../../parquet/')
CUSTOM_CORRECTIONS_PATH = os.path.join(current_file_dir,'../../custom_corrections.csv')

AVAILABLE_DATSETS_PATHS = []#AmazonScience/massive localized datsets available in PARQUET_BASE_DIR

parquet_folder_content = os.listdir(PARQUET_BASE_DIR)
for item in parquet_folder_content:
    path =os.path.join(PARQUET_BASE_DIR,item)
    if os.path.isdir(path):
        AVAILABLE_DATSETS_PATHS.append(item)
#the id of this array is used to map AmazonScience/massive intents
AMAZON_MASSIVE_INTENTS = [
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
INTENT_FEATURES = ClassLabel(names=AMAZON_MASSIVE_INTENTS)

#====================
#       UTILITY
#====================
def map_intent_to_string(example):
    intent_id = example['intent']
    if 0 <= intent_id < len(AMAZON_MASSIVE_INTENTS):
        example['intent']= AMAZON_MASSIVE_INTENTS[intent_id]
    else:
        example['intent'] = 'UNKNOWN'
    return example

def add_play_music_intent_and_enUS_locale(example):
    example['intent']= AMAZON_MASSIVE_INTENTS[45]
    example['locale']='en-US'
    return example
#====================
#       SERVICE
#====================
def get_intents_mapping():
    return AMAZON_MASSIVE_INTENTS

def get_massive_dataset():
    print(f"Loading AmazonScience/massive dataset...")
    datasets_by_split = {'train': [], 'test': [], 'validation': []}
    for path in AVAILABLE_DATSETS_PATHS:
        data_files = {
            'train':os.path.join(PARQUET_BASE_DIR,path,'train','**/*.parquet'),
            'test':os.path.join(PARQUET_BASE_DIR,path,'test','**/*.parquet'),
            'validation':os.path.join(PARQUET_BASE_DIR,path,'validation','**/*.parquet'),
        }

        try:
            localized_dataset = load_dataset('parquet',data_files=data_files)

            for split in ['train','test','validation']:
                if split in localized_dataset:
                    datasets_by_split[split].append(localized_dataset[split])
        except Exception as e:
            print(f"Error loading {path}: {e}")#Temporary handling
        
    massive_dataset = {}
    for split in ['train','test','validation']:
        if datasets_by_split[split]:
            massive_dataset[split] = concatenate_datasets(datasets_by_split[split])
    if not massive_dataset:
        raise ValueError(f"No dataset found")#Temporary handling
        
    massive_dataset = DatasetDict(massive_dataset)
    massive_dataset = massive_dataset.map(
            map_intent_to_string,
            batched=False,
            desc="Conversion ID Intent -> String"
        )
    new_features = massive_dataset['train'].features.copy()
    new_features['intent']=INTENT_FEATURES
    massive_dataset = massive_dataset.cast(
        features=new_features
    )
    return massive_dataset

def get_custom_corrections_dataset():
    print(f"Loading custom_corrections dataset...")
    try:
        custom_corrections = load_dataset('csv',data_files={'train':CUSTOM_CORRECTIONS_PATH})
    except Exception as e:
        print(f"Error:{e}")#Temporary handling
    if not custom_corrections:
        raise ValueError(f"No custom corrections found")##Temporary handling
    custom_corrections = custom_corrections.map(
        add_play_music_intent_and_enUS_locale,
        batched=False
    )
    new_features = custom_corrections['train'].features.copy()
    new_features['intent']=INTENT_FEATURES
    custom_corrections = custom_corrections.cast(features=new_features)
    return custom_corrections['train']

def create_custom_dataset(
        use_custom:bool=True,
        use_massive:bool=True,
        relevant_intents=[],
        relevant_intents_quantity:int=0,
        non_relevant_intents_quantity:int=0
        ):
    print
    custom_corrections_dataset = get_custom_corrections_dataset() if use_custom else None

    massive_dataset = get_massive_dataset()#i load it anyway because i need test and validation from it

    massive_subset = None
    if use_massive and len(relevant_intents)>0:
        print(f"Selecting {relevant_intents_quantity} entires with selected intents and {non_relevant_intents_quantity} entries with different intents...")
        massive_dataset_relevant = massive_dataset['train'].filter(lambda example: example['intent'] in relevant_intents)
        massive_dataset_relevant=massive_dataset_relevant.shuffle(seed=42).select(range(min(relevant_intents_quantity,len(massive_dataset_relevant)))).select_columns(['utt','annot_utt','intent','locale']) #i only need this columns to concatenate it with custom_dataset
        
        massive_dataset_non_relevant = massive_dataset['train'].filter(lambda example: example['intent'] not in relevant_intents)
        massive_dataset_non_relevant=massive_dataset_non_relevant.shuffle(seed=42).select(range(min(non_relevant_intents_quantity,len(massive_dataset_non_relevant)))).select_columns(['utt','annot_utt','intent','locale'])
        
        massive_subset = concatenate_datasets([massive_dataset_relevant,massive_dataset_non_relevant])
    elif use_massive:
        massive_subset = massive_dataset['train'].select_columns(['utt','annot_utt','intent','locale'])

    datasets_to_union = []

    if massive_subset is not None:
        datasets_to_union.append(massive_subset)

    if custom_corrections_dataset is not None:
        datasets_to_union.append(custom_corrections_dataset)

    massive_dataset['train']= concatenate_datasets(datasets_to_union)
    
    return massive_dataset
