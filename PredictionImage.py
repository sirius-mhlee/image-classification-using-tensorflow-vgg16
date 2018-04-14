import sys
import cv2

import numpy as np
import tensorflow as tf

import VGG16 as vgg

def print_result(label_file_path, prob, top):
    synset = [line.strip() for line in open(label_file_path).readlines()]

    argsorted_prob = np.argsort(prob)[::-1]

    for idx in range(top):
        result = synset[argsorted_prob[idx]]
        print(("Result : {0}, Prob : {1}".format(result, prob[argsorted_prob[idx]])))

def main():
    img = cv2.imread(sys.argv[3])
    reshape_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    np_img = np.asarray(reshape_img, dtype=float)
    expand_np_img = np.expand_dims(np_img, axis=0)

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        feed_dict = {images:expand_np_img}

        vgg_model = vgg.VGG16(sys.argv[1], False)
        with tf.name_scope("vgg_content"):
            vgg_model.build(images)

        sess.run(tf.global_variables_initializer())

        prob = sess.run(vgg_model.prob, feed_dict=feed_dict)
        print_result(sys.argv[2], prob[0], 5)

if __name__ == '__main__':
    main()
