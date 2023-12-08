from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
l = keras.layers

print("np.__version__", np.__version__)
print("tf.__version__", tf.__version__)
print("keras.__version__", keras.__version__)

train_inp1 = np.random.randn(10000).reshape(1000, 10, 1)
train_inp2 = np.random.randn(3000).reshape(1000, 3)

train_out1 = np.random.randn(1000)
train_out2 = np.random.randn(3000).reshape(1000, 3)


inp1 = l.Input(shape=(10, 1), name="inp1")
inp2 = l.Input(shape=(3,), name="inp2")
x = l.Conv1D(10, 10)(inp1)
x = l.Flatten()(x)
x = l.Dense(10, activation="relu")(x)
y = l.Dense(10, activation="relu")(inp2)
z = l.Concatenate()([x, y])
out1 = l.Dense(1, activation="relu", name="out1")(z)
out2 = l.Dense(3, activation="relu", name="out2")(z)

model = keras.models.Model([inp1, inp2], [out1, out2])  # {"out1":out1, "out2":  out2}, {"out1":out1, "out2":  out2})

model.compile(loss={"out1": "mse", "out2": "mse"}, optimizer="sgd")

model.fit({"inp1": train_inp1, "inp2": train_inp2}, {"out1": train_out1, "out2": train_out2}, batch_size=10, epochs=2)
model.save("./saved_model")

model_loaded = load_model("./saved_model")
model_loaded.fit({"inp1": train_inp1, "inp2": train_inp2}, {"out1": train_out1, "out2": train_out2}, batch_size=10, epochs=2)
