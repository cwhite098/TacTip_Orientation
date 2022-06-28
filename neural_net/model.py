import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD

def create_network(input_height, input_width, num_outputs):

    model = Sequential()

    # 1st convolution
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',
                    input_shape=(input_height, input_width, 1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 2nd convolution
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 3rd convolution
    model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 4th convolution
    model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 5th convolution
    model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Flatten the convolved images
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(num_outputs))

    model.compile(loss='mean_squared_error', 
                    optimizer=SGD(learning_rate=0.0001), metrics=['accuracy'])

    return model