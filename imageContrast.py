import tensorflow as tf
import numpy as np


# import matplotlib.pyplot as plt

import loadDataset as ds


def contrast_image(img, factor):
    return tf.image.adjust_contrast(img, factor)

'''
argument: 
images: a batch of images with uint8 RGB values
low: lower bound of contrast factor
high: upper bound of contrast factor
output: a batch of images applied contrast with same random value of contrast factor

'''
def contrast(images, low=0.3, high=1):
    sess = tf.InteractiveSession()

    '''
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    '''
    # print (images.shape)

    # contrast factor
    seed = np.random.random_integers(low * 1000, high * 1000)
    contrast_factors = np.full(images.shape[0], seed / 1000.0, np.float32)
    # contrast_factors = np.random.uniform(low, high, images.shape[0]).astype(np.float32)
    # print (contrast_factors[0])
    # print (contrast_factors.shape)

    fn = lambda x: contrast_image(x[0], x[1])
    elems = (images, contrast_factors)
    contrasted_images = tf.map_fn(fn, elems=elems, dtype=tf.uint8)
    result = sess.run(contrasted_images)

    # print (result.shape)
    return result

if __name__ == '__main__':
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    images = contrast(images)
    print (images.shape)

