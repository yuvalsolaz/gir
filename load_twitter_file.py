import os
import sys
import json
import glob
import bz2
import pandas as pd
from tqdm import tqdm
from geojson import GeoJSON
from shapely.geometry import shape
import shapely.wkt as wkt


# from visualization import visualize_tweets

def to_wkt(rec):
    if rec is None:
        return None
    try:
        gj = GeoJSON(rec)
        list.reverse(gj['coordinates'])
        return shape(gj).wkt
    except Exception as ex:
        print('error parsing shape ')
    return None


def load_twitter_file(file):
    with bz2.open(file, 'rb') as fp:
        lines = fp.readlines()
        tweets = []
        for line in lines:
            tweet = json.loads(line)
            tweets.append(tweet)
    df = pd.DataFrame(data=tweets)
    df['wkt'] = df.loc[df.geo.notnull(), 'geo'].apply(lambda t: to_wkt(t))
    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <twitter_files_path>')
        exit(1)

    twitter_directory = sys.argv[1]
    print(f'loading twitter files from {twitter_directory}')

    # All files ending with .json
    files = glob.glob(os.path.join(twitter_directory, '**/*.bz2'), recursive=True)
    tweets_df = pd.DataFrame()
    tweets_count = 0
    for file in tqdm(files, unit=' file'):
        df = load_twitter_file(file)
        tweets_count += df.shape[0]
        location_df = df.loc[df.wkt.notnull()]
        tweets_df = pd.concat([tweets_df, location_df])
        # extract x y from wkt
        tweets_df['lat'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).x)
        tweets_df['lon'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).y)
        print(f' {tweets_df.shape[0]} location tweets out of {tweets_count} tweets')
    output_file = r'./data/twitter/tweets.csv'
    print(f'total:{tweets_df.shape} location tweets downloaded writing to {output_file}')
    tweets_df.to_csv(output_file)

    pass
