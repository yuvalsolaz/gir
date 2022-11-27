import sys
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
    distances_vector = np.array(sorted(df['distance'].values))
    median_error = np.median(distances_vector)
    mean_error = np.mean(distances_vector)
    print(f'Median error: {median_error}')
    print(f'Mean error: {mean_error}')
    log_distances_vector = np.log(np.array(distances_vector) + 1)
    k = np.log(161)
    accuracy_at_161 = np.count_nonzero(log_distances_vector < k) / len(log_distances_vector)
    auc = np.trapz(log_distances_vector) / (np.log(20039000) * (len(log_distances_vector) - 1))
    print(f'Accuracy to 161 km: {accuracy_at_161}')
    print(f'AUC = {auc}')  # Trapezoidal rule.
    print("==============================================================================================")
    return auc


def geo2utm(geo_coord_list):
    utm_coord_list = list()
    for coord in geo_coord_list:
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

    print(f'loading {evaluation_file}...')
    df = pd.read_json(evaluation_file)
    print(f'{df.shape[0]} records loaded')

    tokenizer, model = load_model(checkpoint=checkpoint)
    tqdm.pandas()

    output_file = evaluation_file.replace('.json', '_inference.csv')
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
    df['centrid_distance'] = df.apply(lambda t: get_distance(t, use_centroid=True), axis=1)
    print(f'write {df.shape[0]} records with distances to {output_file}')
    df.to_csv(output_file)
    auc = auc(df)


