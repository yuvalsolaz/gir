import sys
import datashader as ds
import datashader.transfer_functions as tf
import colorcet
import datasets
from datasets import Dataset
import pandas as pd
from geolabel import label_field, cell2geo
# visualize cells



def unique_labels(ds):
    ds.set_format('pandas')
    unique_labels = ds[label_field].unique()
    print(f'{label_field} value counts:\n{ds[label_field].value_counts()}')
    ds.reset_format()
    return unique_labels

def create_train_cells_polygons_file(dataset_path, out_file):
    ds = datasets.load_from_disk(dataset_path=dataset_path)
    print(f'{ds.shape} samples loaded')
    train = ds['train']
    unique_ids = unique_labels(train)
    print(f'{train.shape[0]} train samples {len(unique_ids)} unique labels')
    cells = [cell2geo(cell_id[:-1])[0][0] for cell_id in unique_ids]
    poly_list = [f'POLYGON(({c[1]} {c[0]},{c[3]} {c[2]},{c[5]} {c[4]},{c[7]} {c[6]},{c[9]} {c[8]}))' for c in cells]
    print(f'write train cells polygons to {out_file}')

    with open(out_file, 'w') as fp:
        fp.write('cell_id;wkt\n')
        for i,poly in enumerate(poly_list):
            fp.write(f'{unique_ids[i]};{poly}\n')

    train_df = pd.DataFrame(train)
    train_df['point_wkt'] = train_df.apply(lambda t : f'POINT({t["longitude"]} {t["latitude"]})',axis=1)
    train_df.to_csv(f'points_{out_file}')



def create_data_points_file(data_points_file, out_file):
    df = pd.read_csv(data_points_file, dtype=str)
    return len(df)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <dataset_path>')
        exit(1)
    dataset_path = sys.argv[1]
    out_file = 'train_cells_wkt.csv'
    print(f'loading dataset: {dataset_path}...')
    print(f'writing cells polygons to {out_file}...')
    create_train_cells_polygons_file(dataset_path=dataset_path, out_file=out_file)


def visualize_dataset(dataset:Dataset):
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg_points = cvs.points(source=pd.DataFrame(dataset), x='x', y='y')
    img = ds.tf.shade(agg_points, cmap=colorcet.fire, how='log')
    return img


def visualize_tweets(df, xcol, ycol):
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg = cvs.points(df, x=xcol, y=ycol, agg=ds.count())
    data = tf.shade(agg, cmap=["lightblue", "darkblue"], how='log')
    data = tf.set_background(img=data, color="black")
    data.to_pil().show()

def __visualize_tweets(df, xcol, ycol):
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg = cvs.points(df, x=xcol, y=ycol)
    ds.tf.set_background(ds.tf.shade(agg, cmap=colorcet.fire), "black")
    data = ds.tf.shade(agg)# , cmap=colorcet.fire, how='log')
    img = data.to_pil()
    # img.save('tmp.png')
    img.show()
