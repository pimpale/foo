import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# tf.debugging.set_log_device_placement(True)

INP_SIZE = 10000

# Goal: Train neural network to add numbers


inp_a = keras.layers.Input(shape=(INP_SIZE,), dtype="float32")
inp_b = keras.layers.Input(shape=(INP_SIZE,), dtype="float32")
concat = keras.layers.Concatenate()([inp_a, inp_b])

hidden0 = keras.layers.Dense(1000, activation='relu')(concat);
hidden1 = keras.layers.Dense(1000, activation='relu')(hidden0);
hidden2 = keras.layers.Dense(1000, activation='relu')(hidden1);

output = keras.layers.Dense(1, activation='linear')(hidden2);


model = keras.Model( inputs=[ inp_a, inp_b ], outputs=[ output ] )

def loss_fn(y_true, y_pred):
    print("TRUE")
    print(y_true)
    print("PRED")
    print(y_pred)
    diff = y_true - y_pred
    return diff * diff

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=loss_fn
)

model.summary()

from time import sleep

BATCH_SIZE = 5000


with tf.device('/GPU:0'):
    while True:
        # Come up with some training data
        # 5 length 4 vectors
        v_a = np.random.sample((BATCH_SIZE,INP_SIZE))
        v_b = np.random.sample((BATCH_SIZE,INP_SIZE))

        print("SOURCE DATA:");
        print(v_a, "\n", v_b)

        # Show what model currently predicts
        ret = model.predict(
          [v_a, v_b]
        )
        print("OUTPUT:\n", ret)

        true_batch = np.zeros((BATCH_SIZE, 1))
        for i in range(0, BATCH_SIZE):
            true_batch[i][0] = np.sum(v_a[i])  + np.sum(v_b[i])


        print("TRUE COMPLETION DATA:\n", true_batch)

        model.fit(
          [v_a, v_b],
          true_batch
        )
