import os
import sys
import json
import glob
import bz2
from tqdm import tqdm

def load_twitter_file(file):
    print(f'loading {file}')
    # with open(file) as fp:
    with bz2.open(file,'rb') as fp:
        count = 1
        lines = fp.readlines()
        for line in lines:
            tweet = json.loads(line)
            if 'coordinates' in tweet and tweet['coordinates'] is not None:
                print(f'{count} coordinates:{tweet["coordinates"]} text:{tweet["text"]}')
            if 'geo' in tweet and tweet['geo'] is not None:
                print(f'{count} geo:{tweet["geo"]} text:{tweet["text"]}')
            count += 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <twitter_files_path>')
        exit(1)

    twitter_directory = sys.argv[1]
    print(f'loading twitter files from {twitter_directory}')

    # All files ending with .json
    files = glob.glob(os.path.join(twitter_directory, '**/*.bz2'), recursive=True)

    for file in tqdm(files, unit=' file'):
        load_twitter_file(file)


