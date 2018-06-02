import tensorflow as tf
import numpy as np
import math

# import matplotlib.pyplot as plt

import loadDataset as ds


ROTATE_DEGREE = 30

def rotate_image(img, angle):
    return tf.contrib.image.rotate(img,
                                        angle,
					interpolation='BILINEAR')

'''
argument: a batch of images with uint8 RGB values
output: a batch of rotated (degree by default is 30) images 

'''

def rotate(images):
    sess = tf.InteractiveSession()

    '''
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    '''
    # print (images.shape)

    # convert degree to scalar angle
    
    rotate_degrees = []
    for i in range(images.shape[0]):
        rotate_degrees.append(ROTATE_DEGREE * math.pi / 180)
    rotate_degrees = tf.convert_to_tensor(rotate_degrees)

    # print (crop_values.shape)

    fn = lambda x: rotate_image(x[0], x[1])
    elems = (images, rotate_degrees)
    rotated_images = tf.map_fn(fn, elems=elems, dtype=tf.uint8)
    result = sess.run(rotated_images)

    # print (result.shape)
    return result

if __name__ == '__main__':
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    images = rotate(images)
    print (images.shape)

