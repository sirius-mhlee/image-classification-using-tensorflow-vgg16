import sys
import cv2

import numpy as np
import tensorflow as tf

import VGG16 as vgg
import DataOperator as do

def print_result(label_file_path, prob, top):
    label_file = open(label_file_path)
    synset = [line.strip() for line in label_file.readlines()]
    label_file.close()

    argsorted_prob = np.argsort(prob)[::-1]

    for idx in range(top):
        result = synset[argsorted_prob[idx]]
        print('Result : {0}, Prob : {1}'.format(result, prob[argsorted_prob[idx]]))

def main():
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, 224, 224, 3])

        npy_file, npy_mean = do.load_npy(sys.argv[1], sys.argv[2])

        vgg_model = vgg.VGG16(npy_file, npy_mean, False)
        with tf.name_scope('vgg_content'):
            vgg_model.build(image)

        sess.run(tf.global_variables_initializer())

        img = do.load_image(sys.argv[4])
        feed_dict = {image:img}

        prob = sess.run(vgg_model.prob, feed_dict=feed_dict)
        print_result(sys.argv[3], prob[0], 5)

if __name__ == '__main__':
    main()
