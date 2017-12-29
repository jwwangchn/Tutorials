# 保存 saver

import tensorflow as tf
import numpy as np
#
## 1. 保存
W = tf.Variable([[1,2,3], [3,4,5]], dtype = tf.float32, name = 'weights')
b = tf.Variable([[1,2,3]], dtype = tf.float32, name = 'biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save path: ", save_path)


## 2. 导入
W_restore = tf.Variable(np.arange(6).reshape((2,3)), dtype = tf.float32, name="weights")
b_restore = tf.Variable(np.arange(3).reshape((1,3)), dtype = tf.float32, name="biases")

saver_restore = tf.train.Saver()
with tf.Session() as sess_restore:
    saver_restore.restore(sess_restore, "my_net/save_net.ckpt")
    print("weights: ", sess_restore.run(W_restore))
    print("biases: ", sess_restore.run(b_restore))