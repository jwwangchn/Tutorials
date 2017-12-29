import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)     # 在run的时候才传值, 相当于变量定于? 先占着内存
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1:[7.0], input2:[2.]}))