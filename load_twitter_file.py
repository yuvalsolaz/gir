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
import tqdm
from geojson import GeoJSON
from shapely.geometry import shape
import shapely.wkt as wkt
from zipfile import ZipFile
import tarfile


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
    try:
        with bz2.open(file,'rb') as fp:
            lines = fp.readlines()
            tweets = []
            for line in lines:
                tweet = json.loads(line)
                tweets.append(tweet)
            df = pd.DataFrame(data=tweets)
            df['wkt'] = df.loc[df.geo.notnull(),'geo'].apply(lambda t: to_wkt(t))
            return df.loc[df.wkt.notnull()]
    except Exception as ex:
        print(f'error loading: {file}: {ex}')
        return pd.DataFrame()



# extract zip / tar file in current directory
def extract(twitter_file):
    data_dir = os.path.dirname(twitter_file)
    if twitter_file.endswith('.zip'):
        with ZipFile(twitter_file) as zfile:
            zfile.extractall(data_dir)
    elif twitter_file.endswith('.tar'):
        print(f'unzip {twitter_file} in {data_dir}')
        with tarfile.open(twitter_file) as tfile:
            tfile.extractall(data_dir)
    else:
        print (f'non recognized file format: {twitter_file}')
        return False
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <twitter_zip_file>')
        exit(1)

    twitter_file = sys.argv[1]
    data_dir = os.path.dirname(twitter_file)
    # extract(twitter_file)

    print(f'loading twitter files from {data_dir}')
    # extract all json files
    files = glob.glob(os.path.join(data_dir, '**/*.bz2'), recursive=True)
    print(f'start loading {len(files)} files...')
    tweets_df = pd.DataFrame()
    file_count = 0
    for file in tqdm.gui.tqdm(files, unit=' file',gui=True):
        tweets_df = pd.concat([tweets_df, load_twitter_file(file)])
        # extract x y from wkt
        tweets_df['lat'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).x)
        tweets_df['lon'] = tweets_df.wkt.apply(lambda t: wkt.loads(t).y)
        file_count += 1
        output_file = twitter_file.replace('.tar','.csv').replace('.zip','.csv')
        print(f'{tweets_df.shape[0]} geo tweets from {file_count} files saving to {output_file}')
        tweets_df.to_csv(output_file)

    print(f'this is the end:\ntotal:{tweets_df.shape} geo tweets downloaded')


