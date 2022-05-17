'''
Geographic labeling flow:
    1. load geo text dataset with text and location ( wiki twitter whatever )
    2. go through s2geometry levels in decreasing order starting from level = 0 to level = max-level
    3. for each sample in the dataset calculate cell-id for current level
    4. add columns with cell-id and cell-id level
    5. calculates value counts for each cell-id
    6. freeze cell-ids if number of samples is less then class threshold

    TODO:  batch mapping

'''

import sys
import numpy as np
import datasets
import s2geometry as s2

# parameters:
max_level = 10  # between 0 to 30
min_cell_samples = 1000
test_size = 0.2

'''
 mapping geo coordinates to s2geometry cell
'''


def geo2cell(lat, lon, level):
    try:
        p = s2.S2LatLng.FromDegrees(lat, lon)
        leaf = s2.S2CellId(p)
        cell = leaf.parent(level)
        return cell
    except Exception as ex:
        print(f'geo2cell exception {ex}')
        return None


'''
    freeze cell for all samples with less then minimum cell samples 
'''


def freeze(dataset, min_cell_samples):
    # calculates value counts for each cell-id
    dataset.set_format(type='pandas', columns='cell_id')
    vc = dataset['cell_id'].value_counts()
    dataset.reset_format()

    def freeze(sample):
        return {'freeze': vc[sample['cell_id']] < min_cell_samples}

    return dataset.map(freeze)


'''
    print dataset aggregation 
'''


def summary(dataset, level):
    dataset.set_format('pandas')
    freeze_counts = dataset['freeze'].value_counts()
    label_counts = len(dataset['cell_id'].unique())
    level_counts = dataset['cell_id_level'].value_counts()  # sort by level
    dataset.reset_format()
    print(f'summary for level {level} freeze count:')
    print(freeze_counts)
    print('level counts:')
    print(level_counts)
    print('labels count:')
    print(label_counts)
    #  'TODO: visualize cells on map...'


def label_data(dataset_file):
    # load geo text dataset with text and location ( wiki twitter whatever )
    print(f'loading dataset: {dataset_file}...')
    ds = datasets.load_dataset("csv", data_files={"train": dataset_file}, split='train[:10%]')
    print(f'{ds.shape[0]} samples loaded')

    print('filter samples without coordinates or text')
    ds = ds.filter(lambda x: x['latitude'] is not None and x['longitude'] is not None and x['english_desc'] is not None)
    print(f'{ds.shape[0]} samples with coordinates and text')

    print('add cell id and freeze columns')
    ds = ds.add_column('cell_id', np.full(ds.shape[0], None, dtype=np.float))
    ds = ds.add_column('freeze', np.full(ds.shape[0], False))

    print(f'go through s2geometry levels in decreasing order starting from level=0 to level={max_level}')
    for level in range(0, max_level + 1):

        print('for each sample in the dataset calculate cell-id for current level')

        def get_cell_id(sample):
            if sample['freeze']:
                return {'cell_id': sample['cell_id'],
                        'cell_id_level': sample['cell_id_level']}
            res = {'cell_id_level': level}
            cellid = geo2cell(lat=sample['latitude'], lon=sample['longitude'], level=level)
            res['cell_id'] = np.float(cellid.id()) if cellid else None
            return res

        print(f"get cell-id's for level: {level}")  
        ds = ds.map(get_cell_id, batched=False)

        print(f're calculates freeze column: True if number of samples is less than {min_cell_samples}')
        ds = freeze(ds, min_cell_samples=min_cell_samples)
        summary(dataset=ds, level=level)

    return ds


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset file>')
        exit(1)
    dataset_file = sys.argv[1]
    print(f'labeling: {dataset_file}...')
    dataset = label_data(dataset_file)

    print(f'split dataset train test: {dataset_file}')
    dataset = dataset.train_test_split(test_size=test_size)

    # save to disk
    output_path = dataset_file.replace('.csv', '_labels')
    print(f'save dataset to: {output_path}')
    dataset.save_to_disk(output_path)
