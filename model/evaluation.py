import sys
import pandas as pd

'''
https://github.com/milangritta/Geocoding-with-Map-Vector

'''


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage: python {sys.argv[0]} <evaluation file> <model file>')
        exit(1)
    evaluation_file = sys.argv[1]
    checkpoint = sys.argv[2]

    print(f'loading {evaluation_file}...')
    df = pd.read_json(evaluation_file)
    print(f'{df.shape[0]} records loaded')

