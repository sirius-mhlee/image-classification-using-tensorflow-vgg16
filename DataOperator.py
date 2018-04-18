import cv2
import random as rand

import numpy as np
import tensorflow as tf

def load_image(img_path):
    img = cv2.imread(img_path)
    reshape_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    np_img = np.asarray(reshape_img, dtype=float)
    expand_np_img = np.expand_dims(np_img, axis=0)
    return expand_np_img

def load_train_data(train_data_path):
    train_data = []

    train_file = open(train_data_path)
    all_line = train_file.readlines()
    for line in all_line:
        split_line = line.split(' ')
        train_data.append((split_line[0], int(split_line[1])))

    return train_data

def get_batch_data(train_data, batch_size):
    rand.shuffle(train_data)
    
    image = []
    label = []

    batch_data = train_data[:batch_size]
    for data in batch_data:
        image.append(load_image(data[0]))
        label.append(data[1])

    batch_image = np.concatenate(image)
    batch_label_op = tf.one_hot(label, on_value=1, off_value=0, depth=1000)

    return batch_image, batch_label_op