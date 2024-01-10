from datasets import load_dataset
from datasets import Dataset
from functools import partial
import re
import pandas as pd
import os
import requests

data_path = "data"
imgs_path = data_path + "/imgs"
csv_path = data_path+"/imgs.csv"
classes_to_filter = ["cat","dog","pillow","plane","penguin","tree","car","cup","house","apple"]
max_number = 10

number_of_accurences = [0] * len(classes_to_filter)

csv_exists = os.path.exists(csv_path)

def is_target_class(x,existing_sample_ids):
    if x is None or x["TEXT"] is None: 
        return False

    if x["SAMPLE_ID"] in existing_sample_ids:
        print("sample exists")
        return False
    
    result = False

    for idx, class_name in enumerate(classes_to_filter):
        if re.search(rf'\b{re.escape(class_name)}\b', x["TEXT"]) and number_of_accurences[idx]<max_number:
            result = True
            number_of_accurences[idx]+=1
            

    return result

def add_class_info(x):
    classes = []
    for class_name in classes_to_filter:
        if re.search(rf'\b{re.escape(class_name)}\b', x["TEXT"]):
            classes.append(class_name)
    x["labels"]=classes

def fix_count_after_download_fail(x):
        for idx, class_name in enumerate(classes_to_filter):
            if re.search(rf'\b{re.escape(class_name)}\b', x["TEXT"]):
                number_of_accurences[idx]-=1

def download_img(data_dict):
        # Download the image
    image_url = data_dict["URL"]
    response = requests.get(image_url)
    
    # Save the image to disk
    image_filename = f"{data_dict['SAMPLE_ID']}.jpg"
    image_path = os.path.join(imgs_path, image_filename)
    
    with open(image_path, 'wb') as f:
        f.write(response.content)

    # Update the dataset with the image path
    data_dict["image_path"] = image_path
    data_dict['path_exists'] = True

def fix_count():
    if csv_exists is not True:
        return
    
    old_csv = pd.read_csv(csv_path,converters={"labels": lambda x: x.strip("[]").replace("'","").split(", ")})

    old_csv["path_exists"]  = old_csv["image_path"].apply(lambda x: os.path.exists(x))

    for index, row in old_csv.iterrows():
        if row["path_exists"]==False:
            continue
        
        for label in row["labels"]:
           if  label in classes_to_filter:
               number_of_accurences[classes_to_filter.index(label)]+=1

    old_csv.to_csv(csv_path)
    print("starting from", sum(number_of_accurences))
    print(number_of_accurences)

def get_existing_sample_ids(): 
    if csv_exists:
        old_csv = pd.read_csv(csv_path)
        return list(old_csv["SAMPLE_ID"])
    else:
        return []

def create_dataset():

    fix_count()
    existing_sample_ids = get_existing_sample_ids()

    split = "train" #only split laion has 
    laion_dataset = load_dataset("laion/laion2B-en",streaming=True)[split]

    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    #collect data
    data = []
    for example in laion_dataset:
        print(number_of_accurences)
        if sum(number_of_accurences)>=max_number*len(number_of_accurences):
            break

        if is_target_class(example,existing_sample_ids):
            try:
                download_img(example)
                add_class_info(example)
                data.append(example)
            except:
                fix_count_after_download_fail(example)

    new_data = pd.DataFrame(data=data)
    # Add examples to the dataset
    if csv_exists:
        old_csv = pd.read_csv(csv_path)
        new_data.to_csv("neueszeug.csv")
        new_data = pd.concat([old_csv, new_data], ignore_index=True)
    
    new_data.to_csv(csv_path)

create_dataset()
