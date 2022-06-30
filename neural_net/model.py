import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.regularizers as regularizers

def create_network(input_height, input_width, num_outputs):

    model = Sequential()

    conv_activation = 'elu'

    # 1st convolution
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                    input_shape=(input_height, input_width, 1)))
    model.add(BatchNormalization())
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 2nd convolution
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 3rd convolution
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
      
    # 4th convolution
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # 5th convolution
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Flatten the convolved images
    model.add(Flatten())

    #model.add(Dense(256, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.001))
    model.add(Dense(16, activation='relu', 
                    kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.064)))

    # Output Layer
    model.add(Dropout(rate=0.001))
    model.add(Dense(num_outputs, activation='linear', 
                   kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.064)))

    model.compile(loss='mean_absolute_error', 
                    optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model