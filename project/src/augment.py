import tensorflow as tf
import numpy as np
import cv2
from math import pi


def augment_video(video, session, oper):
    vid, aug, bc_order, brightness, contrast, flip, angle, boxes, indices = oper
    video_len = video.shape[0]

    # Additional code for cropping.
    crop_off_height = np.random.randint(0, 16)
    crop_off_width = np.random.randint(0, 16)
    crop_out_height = 64
    crop_out_width = 64
    boxes_ = np.tile(np.array([crop_off_height, crop_off_width,
                              crop_off_height + crop_out_height, crop_off_width + crop_out_width]),
                     (video_len, 1)) / 80

    # Perform Augmentation
    return session.run(aug, feed_dict={
        vid: video,
        bc_order: bool(np.random.randint(0, 2)),
        brightness: np.random.uniform(0.0, 0.0),
        contrast: np.random.uniform(0.4, 0.6),
        flip: bool(np.random.randint(0, 2)),
        angle: np.random.uniform(-pi/12, pi/12),
        boxes: boxes_,
        indices: np.arange(video_len)
    })


def augment_video_oper(img_size, vid_length):
    video = tf.placeholder(tf.float32, [vid_length, *img_size])
    brightness, contrast, angle, zoom = [tf.placeholder(tf.float32, [])] * 4
    boxes = tf.placeholder(tf.float32, [vid_length, 4])
    indices = tf.placeholder(tf.int32, [vid_length])
    bc_order, flip = [tf.placeholder(tf.bool, [])] * 2
    #shrink_size = tf.constant([80, 80])
    output_size = tf.constant([64, 64])

    oper = video
    #oper = _shrink(oper, output_size=shrink_size)
    #oper = _brightness_and_contrast(oper, bc_order=bc_order, brightness=brightness, contrast=contrast)
    #oper = tf.image.adjust_contrast(oper, contrast_factor=contrast)

    #oper = _rotate(oper, angle=angle)
    oper = _flip_left_right(oper, flip=flip)
    oper = _crop_and_shrink(oper, boxes, indices, output_size)

    return video, oper, bc_order, brightness, contrast, flip, angle, boxes, indices


def format_image(image):
    img = image
    if img.shape[0] != 80:
        img = cv2.resize(image, (80, 80))
        img = cv2.Canny(img, 50, 100)
    img = img[8:-8, 8:-8]  # Middle (64, 64)
    img = img.reshape((64, 64, 1))
    return img


def format_video(video):
    return np.stack([format_image(video[i]) for i in range(video.shape[0])])


def _brightness_and_contrast(video, bc_order, brightness, contrast):
    return tf.cond(bc_order,
                   lambda: tf.image.adjust_brightness(tf.image.adjust_contrast(video, contrast_factor=contrast), delta=brightness),
                   lambda: tf.image.adjust_contrast(tf.image.adjust_brightness(video, delta=brightness), contrast_factor=contrast))


def _flip_left_right(video, flip):
    return tf.cond(flip,
                   lambda: tf.map_fn(lambda x: tf.image.flip_left_right(x), video),
                   lambda: video)


def _rotate(video, angle):
    return tf.contrib.image.rotate(video, angles=angle)


def _crop_and_shrink(video, boxes, indices, output_size):
    return tf.image.crop_and_resize(video,
                                    boxes=boxes,
                                    box_ind=indices,
                                    crop_size=output_size)


def _shrink(video, output_size):
    return tf.image.resize_images(video, size=output_size)
