import datashader as ds
import datashader.transfer_functions as tf
import colorcet
from datasets import Dataset
import s2geometry as s2

# visualize cells
def get_cell_geometries(cell_id):



def visualize_dataset(dataset:Dataset):
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    dataset.set_format('pandas')
    agg_points = cvs.points(dataset, 'longitude', 'latitude')
    img = ds.tf.shade(agg_points, cmap=colorcet.fire, how='log')
    dataset.reset_format()
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


