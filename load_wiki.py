
"""Get Wikidata dump records as a JSON stream (one JSON object per line)"""
# Modified script taken from this link: "https://www.reddit.com/r/LanguageTechnology/comments/7wc2oi/does_anyone_know_a_good_python_library_code/dtzsh2j/"

import bz2
import json
import pandas as pd
import pydash

i = 0
# an empty dataframe which will save items information
# you need to modify the columns in this data frame to save your modified data
df_record_all = pd.DataFrame(columns=['id', 'type', 'english_label', 'longitude', 'latitude', 'english_desc'])

def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


def save(df, file):
    print(f'saving to {file}')
    df.to_csv(path_or_buf=file)
    print('done saving')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        'dumpfile',
        help=(
            'a Wikidata dumpfile from: '
            'https://dumps.wikimedia.org/wikidatawiki/entities/'
            'latest-all.json.bz2'
        )
    )
    args = parser.parse_args()
    df_record_all = pd.DataFrame(columns=['id', 'type', 'english_label', 'longitude', 'latitude', 'english_desc'])
    for record in wikidata(args.dumpfile):
        # only extract items with geographical coordinates (P625)
        if pydash.has(record, 'claims.P625'):
            print(f'{i} item {record["id"]}')
            latitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.latitude')
            longitude = pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.longitude')
            english_label = pydash.get(record, 'labels.en.value')
            item_id = pydash.get(record, 'id')
            item_type = pydash.get(record, 'type')
            english_desc = pydash.get(record, 'descriptions.en.value')
            df_record = pd.DataFrame({'id': item_id, 'type': item_type, 'english_label': english_label, 'longitude': longitude, 'latitude': latitude, 'english_desc': english_desc}, index=[i])
            df_record_all = df_record_all.append(df_record, ignore_index=True)
            i += 1
            if (i % 50000 == 0):
                save(df_record_all, r'data/extracted/all_items.csv')
    save(df_record_all, r'data/all_items.csv')
    print('All items finished, final CSV exported!')


