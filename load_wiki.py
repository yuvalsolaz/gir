
"""Get Wikidata dump records as a JSON stream (one JSON object per line)"""
# Modified script taken from this link: "https://www.reddit.com/r/LanguageTechnology/comments/7wc2oi/does_anyone_know_a_good_python_library_code/dtzsh2j/"

import bz2
import json
import pandas as pd
import pydash

'''
    wikidata dump file iterator  
'''
def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue

'''
    save data to file TODO: save as hdf 
'''
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
    # df_record_all = pd.DataFrame(columns=['id', 'type', 'english_label', 'longitude', 'latitude', 'english_desc'])
    total_records_counter = 0
    rows = [] # rows collector
    i = 0 # internal geo data rows index
    for record in wikidata(args.dumpfile):
        # only extract items with geographical coordinates (P625)
        total_records_counter += 1
        if pydash.has(record, 'claims.P625'):
            i += 1
            if (i % 100 == 0):
                print(f'{i} geo items from total {total_records_counter} {i*100.0/total_records_counter} percent')

            rows.append( {
            'id': i,
            'item_id': pydash.get(record, 'id'),
            'item_type': pydash.get(record, 'type'),
            'latitude': pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.latitude') ,
            'longitude': pydash.get(record, 'claims.P625[0].mainsnak.datavalue.value.longitude'),
            'english_label': pydash.get(record, 'labels.en.value'),
            'ar_label': pydash.get(record, 'labels.ar.value'),
            'english_desc': pydash.get(record, 'descriptions.en.value'),
            'heb_desc': pydash.get(record, 'descriptions.he.value'),
            'ar_desc': pydash.get(record, 'descriptions.ar.value')
            })

            if (i % 50000 == 0):
                df_record = pd.DataFrame(rows)
                save(df_record, r'data/extracted/all_items.csv')

    df_record = pd.DataFrame(rows)
    save(df_record, r'data/all_items.csv')
    print('All items finished, final CSV exported!')


