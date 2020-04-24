import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

# @tf.function
# def simple_nn_layer(x, y):
#     return tf.nn.relu(tf.matmul(x, y))

# x = tf.random.uniform((3,3))
# y = tf.random.uniform((3,3))

# print(simple_nn_layer(x,y))
# # print(tf.nn.relu(tf.matmul(x, y)))

# def linear_layer(x):
#     return 2 * x + 1

# @tf.function
# def deep_net(x):
#     return tf.nn.relu(linear_layer(x))

# print(deep_net(tf.constant((1,2,3))))

# @tf.function
# def square_if_positive(x):
#     if x > 0:
#         return x * x
#     else:
#         return x

# print(square_if_positive(tf.constant(2)))
# print(square_if_positive(tf.constant(-2)))

class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        print('model made')

    @tf.function
    def call(self, input_data):
        tf.print('@here ', input_data)
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2

model = tf.keras.Sequential([
    CustomModel()
])

print(model(tf.constant([2,4,-9])))