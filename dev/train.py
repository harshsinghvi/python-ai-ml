a = 10
b = 3*10+3
print(b)

import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], dtype=float)

y = x*3+3

print('x=', x)
print('y=', y)

neural_network_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])

neural_network_model.compile(optimizer='adam',loss='mse')
neural_network_model.fit(x,y,epochs=5000)
neural_network_model.save("mymodel1.h5")