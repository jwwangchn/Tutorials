# tensorflow 中的变量

import tensorflow as tf
import numpy as np

state = tf.Variable(0, name = 'counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)



init = tf.initialize_all_variables() # must have if define variable, 所有变量的初始化

with tf.Session() as sess:
    sess.run(init)  # 初始化所有变量
    for _ in range(3):
        sess.run(update)    # 没运行一次, 变量执行一次 add 运行并赋值
        print(sess.run(state))
