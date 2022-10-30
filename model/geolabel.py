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
import s2sphere as s2s

transform = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857")

# parameters:
label_field = 's2_sequence'
max_level = 8  # between 0 to 30
min_cell_samples = 5000
test_size = 0.2

'''
 mapping geo cell string to cell id
'''


def cellid_mapping(max_level=max_level):
    cellio = {}
    for level in range(max_level + 2):
        print(f'cellid mapping level: {level}')
        cc = s2.S2CellId.Begin(level)
        while cc != cc.End(level):
            key = str(cc).replace('/', '').replace('\x00', '')
            cellio[key] = cc.id()
            cc = cc.next()
    return cellio


cellio = cellid_mapping(max_level)

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

#region rectangles
def level2geo(min_level, max_level):
    rects = []
    for level in range(min_level, max_level):
        curr_cell = s2.S2CellId.Begin(level)
        while curr_cell != s2.S2CellId.End(level):
            rects.append(get_cell_rectangle(curr_cell))
            curr_cell = curr_cell.next()
    return rects, 0.0


def get_token_rects(cell_id_token):
    try:
        cellid = cellio.get(cell_id_token, None)  # s2.S2CellId.FromToken(cell_id_token,len(cell_id_token))
        if not cellid:
            print(f'cell {cell_id_token} not in cells dictionary')
            return None, None

        # get area :
        s2cellid = s2.S2CellId(cellid)
        area = s2.S2Cell(s2cellid).ExactArea() * 1e5
        # get all levels cells bounding boxes
        rects = []
        s2scellid = s2s.CellId(cellid)
        while s2scellid.level() > 0:
            rects.append(get_cell_rectangle(s2scellid.id()))
            s2scellid = s2scellid.parent()
        return rects, area
    except Exception as ex:
        print(f'get_token_rects exception {ex}')
        return None, None
#endregion

def get_cell_rectangle(cell_id):
    vertices = []
    s2scell_id = s2s.CellId(cell_id)
    cell = s2s.Cell(s2scell_id)
    rectbound = cell.get_rect_bound()
    for v in [0, 1, 2, 3, 0]:
        vertex = rectbound.get_vertex(v)
        vertices.append(vertex.lat().degrees)
        vertices.append(vertex.lng().degrees)
    return vertices

#region polygon
def get_token_polygons(cell_id_token):
    try:
        cellid = cellio.get(cell_id_token, None)  # s2.S2CellId.FromToken(cell_id_token,len(cell_id_token))
        if not cellid:
            print(f'cell {cell_id_token} not in cells dictionary')
            return None, None
            # get area :
        s2cellid = s2.S2CellId(cellid)
        area = s2.S2Cell(s2cellid).ExactArea() * 1e5
        polygons = []
        s2scellid = s2s.CellId(cellid)
        while s2scellid.level() > 0:
            polygons.append(get_cell_polygon(s2scellid.id()))
            s2scellid = s2scellid.parent()
        return polygons, area



        return get_cell_polygon(s2scellid.id())
    except Exception as ex:
        print(f'cell_id_token2geo exception {ex}')
        return None

def get_token_polygon(cell_id_token):
    try:
        cellid = cellio.get(cell_id_token, None)  # s2.S2CellId.FromToken(cell_id_token,len(cell_id_token))
        if not cellid:
            print(f'cell {cell_id_token} not in cells dictionary')
            return None, None

        s2scellid = s2s.CellId(cellid)
        return get_cell_polygon(s2scellid.id())
    except Exception as ex:
        print(f'cell_id_token2geo exception {ex}')
        return None

def get_cell_polygon(cell_id):
    vertices = []
    cell = s2.S2Cell(s2.S2CellId(cell_id))
    for i in range(0, 4):
        vertex = cell.GetVertex(i)
        latlng = s2.S2LatLng(vertex)
        vertices.append((latlng.lat().degrees(),
                         latlng.lng().degrees()))
    return vertices

'''
    freeze cell for all samples with less then minimum cell samples 
'''
#endregion

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
    label_counts = len(dataset[label_field].unique())
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
        x, y = transform.transform(lat, lon)
        if x == float('inf') or y == float('inf'):
            return {'x': None, 'y': None}
        return {'x': x, 'y': y}
    except Exception as ex:
        print(f'error in conv2webmercator: {ex}')
        return {'x': None, 'y': None}


def label_one_level(ds, level):
    print(f'calculate cell-id for each sample in level {level}')

    def get_cell_id(sample):
        if sample['freeze']:
            return {'cell_id': sample['cell_id'],
                    'cell_id_level': sample['cell_id_level']}
        res = {'cell_id_level': level}
        cellid = geo2cell(lat=sample['latitude'], lon=sample['longitude'], level=level)
        # TODO: res['child_positions'] = ''.join([cellid.child_position(l) for l in range(1,level)])
        #       child_position not implemented in python
        if cellid:
            res['cell_id'] = cellid.ToToken()
            res['face'] = cellid.face()
            res[label_field] = f'{str(cellid)[0]}{str(cellid)[2:]}'
            res['rect'] = get_cell_rectangle(cell_id=cellid)
        return res

    print(f"get cell-id's for level: {level}")
    ds = ds.map(get_cell_id, batched=False)

    print(f'freeze cells with number of samples less than {min_cell_samples}')
    return freeze(ds, min_cell_samples=min_cell_samples)


def label_data(dataset):
    print('filter samples without coordinates or text')
    dataset = dataset.filter(
        lambda x: x['latitude'] is not None and x['longitude'] is not None and x['english_desc'] is not None)
    print(f'{dataset.shape[0]} samples with coordinates and text')

    print('add cell id and freeze columns')
    dataset = dataset.add_column('cell_id', np.full(dataset.shape[0], '', dtype=str))
    dataset = dataset.add_column('freeze', np.full(dataset.shape[0], False))
    print('convert coordinates to web mercator for visualization')
    try:
        dataset = dataset.map(conv2webmercator, batched=False)
    except Exception as ex:
        print(f'error mapping conv : {ex}')
    print(f'go through s2geometry levels in decreasing order starting from level=0 to level={max_level}')
    for level in range(0, max_level + 1):
        dataset = label_one_level(dataset, level)
        if dataset:
            summary(dataset=dataset, level=level)

    return dataset.filter(lambda x: x['cell_id'] is not None)


#
# def map_labels(ds):
#     print('label mapping...')
#     ds.set_format('pandas')
#     labels = ds['cell_id'].unique()
#     ds.reset_format()
#     label2id = {k: np.where(labels == k)[0][0] for k in labels}
#
#     def token2id(sample):
#         return {'id_labels': label2id[sample['cell_id']]}
#
#     ds = ds.map(token2id)
#     return ds


transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset file>')
        exit(1)

    dataset_file = sys.argv[1]

    # load geo text dataset with text and location ( wiki twitter whatever )
    print(f'loading dataset: {dataset_file}...')
    dataset = datasets.load_dataset("csv", data_files={"train": dataset_file}, split='train[:1%]')
    print(f'{dataset.shape[0]} samples loaded')

    dataset = label_data(dataset=dataset)
    print(f'{dataset.shape[0]} labeled samples')

    # print(f'label mapping...')
    # dataset = map_labels(dataset)

    print(f'split dataset train test: {dataset_file}')
    dataset = dataset.train_test_split(test_size=test_size)

    # save to disk
    output_path = dataset_file.replace('.csv', '_labels')
    print(f'save dataset to: {output_path}')
    dataset.save_to_disk(output_path)
