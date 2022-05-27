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
import pyproj

transform = pyproj.Transformer.from_crs("epsg:4326","epsg:3857")

# parameters:
max_level = 2  # between 0 to 30
min_cell_samples = 50000
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
    level_counts = dataset['cell_id_level'].value_counts().sort_index(ascending=True)
    dataset.reset_format()
    print(f'summary for level {level} freeze count:')
    print(freeze_counts)
    print('level counts:')
    print(level_counts)
    print('labels count:')
    print(label_counts)



def conv2webmercator(sample):
    try:
        lat = sample['latitude']
        lon = sample['longitude']
        x,y = transform.transform(lat,lon)
        if x == float('inf') or y == float('inf'):
            return {'x':None,'y':None}
        return {'x':x,'y':y}
    except Exception as ex:
        print(f'error in conv2webmercator: {ex}')
        return {'x':None,'y':None}

def get_cell_rectangle(cell_id):
    cell = s2.S2Cell(cell_id)
    r = cell.GetRectBound()
    # convert rectangle coordinates to web mercator
    x_hi, y_hi = transform.transform(r.lat_hi().degrees(), r.lng_hi().degrees())
    x_lo, y_lo = transform.transform(r.lat_lo().degrees(), r.lng_lo().degrees())
    if x_lo == float('inf') or y_lo == float('inf') or x_hi == float('inf') or y_hi == float('inf'):
        return None
    else:
        return [x_lo, y_hi, x_hi, y_hi, x_hi, y_lo, x_lo, y_lo, x_lo, y_hi]


def label_one_level(ds, level):
    print(f'calculate cell-id for each sample in level {level}')
    def get_cell_id(sample):
        if sample['freeze']:
            return {'cell_id': sample['cell_id'],
                    'cell_id_level': sample['cell_id_level']}
        res = {'cell_id_level': level}
        cellid = geo2cell(lat=sample['latitude'], lon=sample['longitude'], level=level)
        if cellid:
            res['cell_id'] = cellid.ToToken()
            res['rect'] = get_cell_rectangle(cell_id=cellid)
        return res

    print(f"get cell-id's for level: {level}")
    ds = ds.map(get_cell_id, batched=False)

    print(f'freeze cells with number of samples less than {min_cell_samples}')
    return freeze(ds, min_cell_samples=min_cell_samples)


def label_data(dataset_file):
    # load geo text dataset with text and location ( wiki twitter whatever )
    print(f'loading dataset: {dataset_file}...')
    ds = datasets.load_dataset("csv", data_files={"train": dataset_file}, split='train[:2%]')
    print(f'{ds.shape[0]} samples loaded')

    print('filter samples without coordinates or text')
    ds = ds.filter(lambda x: x['latitude'] is not None and x['longitude'] is not None and x['english_desc'] is not None)
    print(f'{ds.shape[0]} samples with coordinates and text')

    print('add cell id and freeze columns')
    ds = ds.add_column('cell_id', np.full(ds.shape[0], '', dtype=str))
    ds = ds.add_column('freeze', np.full(ds.shape[0], False))
    print('convert coordinates to web mercator for visualization')
    try:
        ds = ds.map(conv2webmercator,batched=False)
    except Exception as ex:
        print (f'error mapping conv : {ex}')
    print(f'go through s2geometry levels in decreasing order starting from level=0 to level={max_level}')
    for level in range(0, max_level + 1):
        ds = label_one_level(ds, level)
        if ds:
            summary(dataset=ds, level=level)

    return ds.filter(lambda x: x['cell_id'] is not None)


def map_labels(ds):
    print('label mapping...')
    ds.set_format('pandas')
    labels = ds['cell_id'].unique()
    ds.reset_format()
    label2id = {k: np.where(labels == k)[0][0] for k in labels}

    def token2id(sample):
        return {'labels': label2id[sample['cell_id']]}

    ds = ds.map(token2id)
    return ds

transformer = pyproj.Transformer.from_crs("epsg:4326","epsg:3857")

def conv2webmercator(sample):
    try:
        lat = sample['latitude']
        lon = sample['longitude']
        x,y = transformer.transform(lat,lon)
        if x == float('inf') or y == float('inf'):
            return {'x':None,'y':None}
        return {'x':x,'y':y}
    except Exception as ex:
        print(f'error in conv2webmercator: {ex}')
        return {'x':None,'y':None}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset file>')
        exit(1)
    dataset_file = sys.argv[1]
    print(f'labeling: {dataset_file}...')
    dataset = label_data(dataset_file)
    print(f'{dataset.shape[0]} labeled samples')

    print(f'label mapping...')
    dataset = map_labels(dataset)

    print(f'split dataset train test: {dataset_file}')
    dataset = dataset.train_test_split(test_size=test_size)

    # save to disk
    output_path = dataset_file.replace('.csv', '_labels')
    print(f'save dataset to: {output_path}')
    dataset.save_to_disk(output_path)
