import tensorflow as tf
import numpy as np
import os
import cv2
from math import pi


def batch_generator(X, Y, session, batch_size=64, vid_length=45, shuffle=False):
    N, H, W, R = X.shape
    total_length = N - vid_length

    order = np.random.permutation(total_length) if shuffle is True else np.arange(total_length)
    oper = _augment_video_oper(img_size=X.shape[1:], vid_length=vid_length)

    for i in range(int(total_length / batch_size)):
        start_indices = order[(i)*batch_size:(i+1)*batch_size]
        x_batch = [X[(start):(start+vid_length)] for start in start_indices]
        y_batch = [Y[(start):(start+vid_length)] for start in start_indices]

        for j, x_vid in enumerate(x_batch):
            x_batch[j] = _augment_video(x_vid, session, oper)

        yield np.stack(x_batch), np.stack(y_batch)


def batch_generator_from_path(img_paths, csv_paths, session, batch_size=64, vid_length=45, shuffle=False):
    if not isinstance(img_paths, (tuple, list)):
        img_paths = [img_paths]
    if not isinstance(csv_paths, (tuple, list)):
        csv_paths = [csv_paths]

    csv_files = np.concatenate([_csv_to_list(path) for path in csv_paths])
    img_files = [[path + "/" + str(i) + ".jpeg" for i in range(1, len(csv_files[j]) + 1)] for j, path in enumerate(img_paths)]
    img_files = [item for sublist in img_files for item in sublist]  # Flatten to 1D
    total_length = sum(len(lst) for lst in csv_files) - vid_length  # Number of video starting points

    image_shape = _load_images(img_files[0]).shape
    print(image_shape)
    oper = _augment_video_oper(img_size=image_shape, vid_length=vid_length)
    order = np.random.permutation(total_length) if shuffle is True else np.arange(total_length)

    for i in range(int(total_length / batch_size)):
        start_indices = order[(i)*batch_size:(i+1)*batch_size]
        x_batch = [_load_images(img_files[(start):(start+vid_length)]) for start in start_indices]
        y_batch = [csv_files[(start):(start+vid_length)] for start in start_indices]

        for j, x_vid in enumerate(x_batch):
            x_batch[j] = _augment_video(x_vid, session, oper)

        yield np.stack(x_batch), np.stack(y_batch)


def _csv_to_list(path):
    with open(path) as f:
        return f.read().splitlines()


def _load_images(paths: str or list or tuple) -> np.array:
    if isinstance(paths, str):
        paths = [paths]
    imgs = np.array([cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for path in paths]).astype(float) / 255
    print(paths, imgs.shape)
    return imgs if imgs.shape[0] > 1 else imgs.reshape(imgs.shape[1:])


def _augment_video(video, session, oper):
    vid, aug, bc_order, brightness, contrast, flip, angle, boxes, indices = oper
    image_size = video.shape[1:3]
    video_len = video.shape[0]

    # Additional code for cropping.
    crop_off_height = np.random.randint(0, int(128 * 0.2))
    crop_off_width = np.random.randint(0, int(128 * 0.2))
    crop_out_height = int(128 * 0.8)
    crop_out_width = int(128 * 0.8)
    boxes_ = np.tile(np.array([crop_off_height, crop_off_width,
                              crop_off_height + crop_out_height, crop_off_width + crop_out_width]), (video_len, 1)) / 128

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


def _augment_video_oper(img_size, vid_length):
    video = tf.placeholder(tf.float32, [vid_length, *img_size])
    brightness, contrast, angle, zoom = [tf.placeholder(tf.float32, [])] * 4
    boxes = tf.placeholder(tf.float32, [vid_length, 4])
    indices = tf.placeholder(tf.int32, [vid_length])
    bc_order, flip = [tf.placeholder(tf.bool, [])] * 2
    shrink_size = tf.constant([128, 128])
    output_size = tf.constant([64, 64])

    oper = video
    oper = _shrink(oper, output_size=shrink_size)
    #oper = _brightness_and_contrast(oper, bc_order=bc_order, brightness=brightness, contrast=contrast)
    #oper = tf.image.adjust_contrast(oper, contrast_factor=contrast)
    oper = _flip_left_right(oper, flip=flip)
    oper = _rotate(oper, angle=angle)
    oper = _crop_and_shrink(oper, boxes, indices, output_size)

    return video, oper, bc_order, brightness, contrast, flip, angle, boxes, indices


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
