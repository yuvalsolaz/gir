
import os
import sys
import json
import glob
import bz2
import pandas as pd
from tqdm import tqdm
from geojson import GeoJSON
from shapely.geometry import shape

def to_wkt(rec):
    if rec is None:
        return None
    try:
        return shape(GeoJSON(rec).wkt)
    except Exception as ex:
        print('error parsing shape ')
    return None

def load_twitter_file(file):
    with bz2.open(file,'rb') as fp:
        lines = fp.readlines()
        tweets = []
        for line in lines:
            tweet = json.loads(line)
            tweets.append(tweet)
    df = pd.DataFrame(data=tweets)
    gdf = df[df.coordinates.notnull()].apply(lambda t: to_wkt(t['geo']))
    return gdf



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <twitter_files_path>')
        exit(1)

    twitter_directory = sys.argv[1]
    print(f'loading twitter files from {twitter_directory}')

    # All files ending with .json
    files = glob.glob(os.path.join(twitter_directory, '**/*.bz2'), recursive=True)
    tweets_df = pd.DataFrame()
    for file in tqdm(files, unit=' file'):
        tweets_df = pd.concat([tweets_df, load_twitter_file(file)])

    print(f'total:{tweets_df.shape} tweets downloaded' )
    tweets_df.to_hdf(r'tweets.h5',key='obama')
    tweets_geo_df = tweets_df[tweets_df.coordinates.notnull()]



