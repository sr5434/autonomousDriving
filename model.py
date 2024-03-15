import tensorflow as tf
import keras
from keras import layers

class CarModel(keras.Model):

    def __init__(self):
        super().__init__()
        #Nvidia self driving car implementation

        #Convolutional component
        self.conv_comp = keras.Sequential(
            [
                layers.Conv2D(24, 5, strides=2, data_format='channels_last', activation='elu'),
                layers.Conv2D(36, 5, strides=2, data_format='channels_last', activation='elu'),
                layers.Conv2D(48, 5, strides=2, data_format='channels_last', activation='elu'),
                layers.Conv2D(64, 3, data_format='channels_last', activation='elu'),
                layers.Conv2D(64, 3, data_format='channels_last', activation='elu'),
                layers.Dropout(0.5),
                layers.Flatten()
            ],
            name="convolutional_component"
        )
        #Individual control heads

        #Control steering
        self.fc_component = keras.Sequential(
            [
                layers.Dense(1164, activation='elu'),
                layers.Dense(100, activation='elu'),
                layers.Dense(50, activation='elu'),
                layers.Dense(10, activation='elu'),
                layers.Dense(1)
            ],
            name="fully_connected_component"
        )
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], 3, 70, 320])#Reshape the input
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        #Run through convolutional layers
        x = self.conv_comp(inputs)
        #Reshape
        #Run thrugh fully connected layers
        outputs = self.fc_component(x)
        return outputs