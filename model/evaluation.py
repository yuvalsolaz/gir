import sys
import os
import numpy as np
import pandas as pd
from shapely import geometry
import shapely.wkt as wkt
import pyproj
from tqdm import tqdm
from model.inference import load_model, inference
from model.geolabel import get_token_polygon

transform = pyproj.Transformer.from_crs("epsg:4326", "epsg:32636")


def get_distance(t, use_centroid=False):
    infer_polygon = wkt.loads(t['inference_polygon'])
    gt_loc = wkt.loads(t['gt_location'])
    return infer_polygon.centroid.distance(gt_loc) if use_centroid else infer_polygon.distance(gt_loc)


'''
Area Under the Curve (AUC) calculates the area under the curve of the distribution of geocoding error distances.
A geocoding system is better if the area under the curve is smaller.
Formally: AUC = ln (ActualErrorDistance) / MaxPossibleErrors
where ActualErrorDistance is the area under the curve, and MaxPossibleErrors is the farthest distance between two places
on earth. 
The value of AUC is between 0 and 1 and the difference between two small errors (such as 10 and 20 km) is more 
significant than the same difference between two large errors (such as 110 and 120 km).
This makes AUC more popular than Accuracy@k
km/miles (Jurgens et al., 2015).
'''

def auc(df):
    """
    Prints Mean, Median, AUC and acc@161km for the list.
    :param accuracy: a list of geocoding errors
    """
    distances_vector = np.array(sorted(df['centroid_distance'].values.astype(np.float)))
    distances_vector = distances_vector[distances_vector < 1e100]

    median_error = np.median(distances_vector)
    mean_error = np.mean(distances_vector)
    print(f'Median error: {int(median_error/1000.0)} km')
    print(f'Mean error: {int(mean_error/1000.0)} km')

    k = 161000 # for accuracy @161
    accuracy_at_161 = np.count_nonzero(distances_vector < k) / len(distances_vector)
    print(f'Accuracy@161 km: {accuracy_at_161}')

    log_distances_vector = np.log(np.array(distances_vector) + 1)
    auc = np.trapz(log_distances_vector) / (np.log(20039000) * (len(log_distances_vector) - 1))
    print(f'AUC = {auc}')  # Trapezoidal rule.
    return auc


def geo2utm(geo_coord_list):
    utm_coord_list = list()
    for coord in geo_coord_list:
        if coord is None:
            continue
        lat = coord[0]
        lon = coord[1]
        x, y = transform.transform(lat, lon)
        utm_coord_list.append((x,y))
    return utm_coord_list


'''
https://github.com/milangritta/Geocoding-with-Map-Vector
'''


def get_sentence_polygon(text):
    cellid, score = inference(tokenizer=tokenizer, model=model, sentence=text[:512])
    inference_poly = get_token_polygon(cell_id_token=cellid)
    utm_inference_poly = geo2utm(inference_poly)
    return geometry.Polygon(utm_inference_poly).wkt


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage: python {sys.argv[0]} <evaluation file> <model file>')
        exit(1)
    evaluation_file = sys.argv[1]
    checkpoint = sys.argv[2]

    output_file = evaluation_file.replace('.json', '_inference.csv')
    if os.path.exists(output_file):
        print(f'output file exists loading it {output_file}...')
        df = pd.read_csv(output_file)
        auc(df)
        exit(0)

    print(f'loading {evaluation_file}...')
    df = pd.read_json(evaluation_file)
    print(f'{df.shape[0]} records loaded')

    tokenizer, model = load_model(checkpoint=checkpoint)
    tqdm.pandas()


    print('inference polygons....')
    df['inference_polygon'] = df.progress_apply(lambda t: get_sentence_polygon(t['text']), axis=1)
    print(f'write {df.shape[0]} records with inference polygon to {output_file}')
    df.to_csv(output_file)

    print('convert gt locations from geo to utm....')
    df['gt_location'] = df.progress_apply(lambda t: geometry.Point(geo2utm([[t['lat'], t['lon']]])).wkt, axis=1)
    print(f'write {df.shape[0]} records with utm locations to {output_file}')
    df.to_csv(output_file)

    print('calculates distance and centroid distance from inference polygon to gt location....')
    df['distance'] = df.apply(lambda t: get_distance(t, use_centroid=False), axis=1)
    df['centroid_distance'] = df.apply(lambda t: get_distance(t, use_centroid=True), axis=1)
    print(f'write {df.shape[0]} records with distances to {output_file}')
    df.to_csv(output_file)
    auc = auc(df)


