import tensorflow as tf
import numpy as np

# 创建训练数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# 创建 tensorflow 结构

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))        # 平方误差
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 学习速率
train = optimizer.minimize(loss)                    # 训练器

init = tf.initialize_all_variables()                # 初始化所有变量


# 训练模型
sess = tf.Session()     # 创建 sess
sess.run(init)          # 初始化

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
