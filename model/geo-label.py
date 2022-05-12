import datasets
from s2grid import geo2cell


'''
Geographic labeling flow:
    1. load geo text dataset with text and location ( wiki twitter whatever )
    2. go through s2geometry levels in decreasing order starting from level = 0 to level = max-level 
    3. for each sample in the dataset calculate cell-id for current level 
    4. add columns with cell-id and cell-id level  
    5. calculates value counts for each cell-id 
    6. freeze cell-ids if number of samples is less then class threshold
'''

# parameters:
max_level = 4 # between 0 to 30

def label_data(dataset_file):
    # load geo text dataset with text and location ( wiki twitter whatever )
    print(f'loading dataset: {dataset_file}...')
    dataset = datasets.load_dataset("csv", data_files={"train": dataset_file}, split='train[:1%]')
    print(f'{dataset.shape[0]} samples loaded')

    print('filter out none')
    dataset = dataset.filter(lambda x: x['latitude'] is not None and x['longitude'] is not None)
    print(f'{dataset.shape[0]} not none samples')
    print(f'go through s2geometry levels in decreasing order starting from level = 0 to level = {max_level}')
    for level in range(0, max_level):
        # for each sample in the dataset calculate cell-id for current level
        def get_cell_id(sample):
            cellid = geo2cell(lat=sample['latitude'], lon=sample['longitude'], level=level)
            if cellid:
                return {'cell_id_level':level,'cell_id': cellid.ToToken()}
            return {'cell_id_level':level,'cell_id': None}

        print('add columns with cell-id and cell-id level')
        dataset = dataset.map(get_cell_id, batched=False)
        # calculates value counts for each cell-id
        # freeze cell-ids if number of samples is less then class threshold
    dataset.save_to_disk(dataset_file.replace('.csv',''))
    return dataset

import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset file>')
        exit(1)
    dataset_file = sys.argv[1]
    print(f'labeling: {dataset_file}...')
    label_data(dataset_file)