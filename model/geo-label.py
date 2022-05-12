import numpy as np
import torch
import transformers
from datasets import load_dataset
from s2grid import geo2cell


'''
Geographic labeling flow:
    1. load geo text dataset with text and location ( wiki twitter whatever )
    2. go through s2geometry levels in decreasing order starting from level = 0 to level = max-level 
    3. for each sample in the dataset calculates cell-id for current level 
    4. add columns with cell-id and cell-id level  
    5. calculates value counts for each cell-id 
    6. freeze cell-ids if number of samples is less then class threshold
'''

# parameters:

def label_data(dataset_file):
    # load geo text dataset with text and location ( wiki twitter whatever )
    dataset = load_dataset("csv", data_files={"train": dataset_file}, delimiter="\t")

    # go through s2geometry levels in decreasing order starting from level = 0 to level = max-level
    for level in range(0, max_level)
    3. for each sample in the dataset calculates cell-id for current level
    4. add columns with cell-id and cell-id level
    5. calculates value counts for each cell-id
    6. freeze cell-ids if number of samples is less then class threshold


def compute_cell_id(sample):
    cell = geo2cell(lat=sample['lat'], lon=sample['lon'], level=0)
    return {f"cell_id_{level}": len(example["review"].split())}
