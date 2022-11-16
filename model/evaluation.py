import sys
import pandas as pd
from shapely import geometry
import pyproj
from tqdm import tqdm
from model.inference import load_model, inference
from model.geolabel import get_token_polygon

transform = pyproj.Transformer.from_crs("epsg:4326", "epsg:32636")

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

#
# def get_label_distance(gt_location, text):
#     inference_poly = get_sentence_polygon(text)
#     utm_inference_poly = geo2utm(inference_poly)
#     utm_gt_location = geo2utm([gt_location])
#     sh_utm_inference_poly = geometry.Polygon(utm_inference_poly)
#     sh_utm_gt_location = geometry.Point(utm_gt_location)
#     distance = sh_utm_inference_poly.distance(sh_utm_gt_location)
#     centroid_distance = sh_utm_inference_poly.centroid.distance(sh_utm_gt_location)
#     print(f'distance:{distance} centroid distance:{centroid_distance} text length:{len(text)}- {text[:100]}')
#     return centroid_distance


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

    df['inference_polygon'] = df.progress_apply(lambda t: get_sentence_polygon(t['text']), axis=1)
    df['gt_point'] = df.progress_apply(lambda t: geo2utm([[t['lat'],t['lon']]]), axis=1)

    def distance(t):
        return geometry.Polygon(t['inference_polygon']).distance(t['gt_location'])

    def centroid_distance(t):
        return geometry.Polygon(t['inference_polygon']).centroid.distance(t['gt_location'])

    df['distance'] = df.apply(lambda t: distance(t), axis=1)
    df['centrid_distance'] = df.apply(lambda t: centroid_distance(t), axis=1)

    output_file = evaluation_file.replace('.json','_inference.csv')
    print(f'write {df.shape[0]} records with error distances to {output_file}')
    df.to_csv(output_file)




