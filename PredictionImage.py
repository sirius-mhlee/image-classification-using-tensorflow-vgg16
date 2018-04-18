import sys
import cv2

import numpy as np
import tensorflow as tf

import VGG16 as vgg
import DataOperator as do

def print_result(label_file_path, prob, top):
    synset = [line.strip() for line in open(label_file_path).readlines()]

    argsorted_prob = np.argsort(prob)[::-1]

    for idx in range(top):
        result = synset[argsorted_prob[idx]]
        print("Result : {0}, Prob : {1}".format(result, prob[argsorted_prob[idx]]))

def main():
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, 224, 224, 3])

        vgg_model = vgg.VGG16(sys.argv[1], False)
        with tf.name_scope("vgg_content"):
            vgg_model.build(image)

        sess.run(tf.global_variables_initializer())

        img = do.load_image(sys.argv[3])
        feed_dict = {image:img}

        prob = sess.run(vgg_model.prob, feed_dict=feed_dict)
        print_result(sys.argv[2], prob[0], 5)

if __name__ == '__main__':
    main()
