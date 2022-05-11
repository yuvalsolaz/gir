'''
functionality to Load parse and filter twitter data from static archive
https://archive.org/details/twitterstream?&sort=-week&page=2

Archive Team: The Twitter Stream Grab
THIS COLLECTION IS NO LONGER UPDATED AND SHOULD BE CONSIDERED A STATIC DATASET.

A simple collection of JSON grabbed from the general twitter stream,
for the purposes of research, history, testing and memory.
This is the "Spritzer" version, the most light and shallow of Twitter grabs.
Unfortunately, we do not currently have access to the Sprinkler or Garden Hose versions of the stream.

'''

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
from zipfile import ZipFile

from visualization import visualize_tweets

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
    with bz2.open(file,'rb') as fp:
        lines = fp.readlines()
        tweets = []
        for line in lines:
            tweet = json.loads(line)
            tweets.append(tweet)
    df = pd.DataFrame(data=tweets)
    df['wkt'] = df.loc[df.geo.notnull(),'geo'].apply(lambda t: to_wkt(t))
    return df.loc[df.wkt.notnull()]



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <twitter_zip_file>')
        exit(1)

    twitter_zip_file = sys.argv[1]
    print(f'loading twitter files from {twitter_zip_file}')

    # extract zip file in current directory
    data_dir = os.path.dirname(twitter_zip_file)
    print(f'unzip {twitter_zip_file} in {data_dir}')
    with ZipFile(twitter_zip_file) as zfile:
        zfile.extractall(data_dir)
    # extract all json files
    files = glob.glob(os.path.join(data_dir, '**/*.bz2'), recursive=True)
    tweets_df = pd.DataFrame()
    for file in tqdm(files, unit=' file'):
        tweets_df = pd.concat([tweets_df, load_twitter_file(file)])
        # extract x y from wkt
        tweets_df['lat'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).x)
        tweets_df['lon'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).y)
        # visualize_tweets(tweets_df,'lat', 'lon')


    print(f'total:{tweets_df.shape} geo tweets downloaded' )
    tweets_df.to_csv(r'./data/twitter/tweets.csv')
    visualize_tweets(tweets_df,'lat', 'lon')
    pass



