import sys
import cv2

import numpy as np
import tensorflow as tf

import VGG16 as vgg
import DataOperator as do

def print_batch_info(epoch_idx, batch_idx, loss_mean_value):
    print('Epoch : {0}, Batch : {1}, Loss Mean : {2}'.format(epoch_idx, batch_idx, loss_mean_value))

def print_epoch_info(epoch_idx, accuracy_mean_value):
    print('Epoch : {0}, Accuracy Mean : {1}'.format(epoch_idx, accuracy_mean_value))

def main():
    max_epoch = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [None, 224, 224, 3])
        label = tf.placeholder(tf.float32, [None, 1000])

        train_data, train_mean = do.load_train_data(sys.argv[1])
        train_size = len(train_data)

        vgg_model = vgg.VGG16(None, train_mean, True)
        with tf.name_scope('vgg_content'):
            vgg_model.build(image)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=vgg_model.fc8, labels=label)
        loss_mean = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss_mean)

        correct_prediction = tf.equal(tf.argmax(vgg_model.fc8, 1), tf.argmax(label, 1))
        accuracy_mean = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        for epoch_idx in range(max_epoch):
            for batch_idx in range(train_size // batch_size):
                batch_image, batch_label = do.get_batch_data(sess, train_data, batch_size)
                feed_dict = {image:batch_image, label:batch_label}

                _, loss_mean_value = sess.run([optimizer, loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_label = do.get_batch_data(sess, train_data, batch_size)
            feed_dict = {image:batch_image, label:batch_label}

            accuracy_mean_value = sess.run(accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        vgg_model.save_npy(sess, sys.argv[4], sys.argv[5])

if __name__ == '__main__':
    main()
