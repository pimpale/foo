import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

INP_SIZE = 4


inp_a = keras.layers.Input(shape=(INP_SIZE,), dtype="float32")
inp_b = keras.layers.Input(shape=(INP_SIZE,), dtype="float32")
output = keras.layers.Add()([inp_a, inp_b])


model = keras.Model( inputs=[ inp_a, inp_b ], outputs=[ output ] )

v_a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
v_b = np.array([[5, 6, 7, 8], [5, 6, 7, 8]])

model.compile()

ret = model.predict(
  [v_a, v_b]
)
print(ret)
