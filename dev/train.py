import tensorflow as tf
import numpy as np


def t(x):
    return x*3+3



x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], dtype=float)

# x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
y = x*3+3
# print(x,y(x))


if __name__ == '__main__':
    neural_network_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
    neural_network_model.compile(optimizer='adam',loss='mse')
    neural_network_model.fit(x,y,epochs=100000)
    neural_network_model.save("linear.model")