import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  # 28x28 of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  #inputlayer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # first hidden layer 128 neurons in layer, activation: stepper function (what makes the neuron fire)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # second hidden layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer (number of classifications, probability function)

model.compile(optimizer= "adam", loss= "sparse_categorical_crossentropy", metrics= ["accuracy"])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)

# plt.imshow(x_train[0])
# plt.show()