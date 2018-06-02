import tensorflow as tf
import numpy as np
import settings

# import matplotlib.pyplot as plt

import loadDataset as ds


def crop_image(img, crop):
    return tf.image.crop_to_bounding_box(img,
                                        crop[0],
                                        crop[1],
                                        crop[2],
                                        crop[3])

'''
argument: a batch of images with uint8 RGB values
output: a batch of cropped (64X64) images taken from the same position of the original images

'''

def crop(images):
    sess = tf.InteractiveSession()

    '''
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    '''
    # print (images.shape)

    # crop value
    offsetHeight = np.random.randint(0, images[0].shape[0]-settings.get_config("resizedH"))
    offsetWidth = np.random.randint(0, images[0].shape[1]-settings.get_config("resizedW"))
    crop_value = [offsetHeight, offsetWidth, settings.get_config("resizedH"), settings.get_config("resizedW")]
    crop_values = []
    for i in range(images.shape[0]):
        crop_values.append(crop_value)
    crop_values = tf.convert_to_tensor(crop_values)

    # print (crop_values.shape)

    fn = lambda x: crop_image(x[0], x[1])
    elems = (images, crop_values)
    cropped_images = tf.map_fn(fn, elems=elems, dtype=tf.uint8)
    result = sess.run(cropped_images)

    # print (result.shape)
    return result

if __name__ == '__main__':
    images, _ = ds.loadDataSet('./dataset/imgs/rock_frames', './dataset/imgs/paper_frames', './dataset/imgs/scissor_frames'
                , './dataset/csvs/rock.csv', './dataset/csvs/paper.csv', './dataset/csvs/scissor.csv')
    images = crop(images)
    print (images.shape)

