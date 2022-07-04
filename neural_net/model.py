from msilib.schema import Class
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.callbacks import EarlyStopping


class PoseNet():
    def __init__(self):
        self.model = None
        self.history = None

    def create_network(self, input_height, input_width, num_outputs):
        '''
        Create the CNN model.
        '''
        self.model = Sequential()

        # Model Hyperparameters
        conv_activation = 'elu'
        dropout_rate = 0.001
        l1_rate = 0.001
        l2_rate = 0.064

        # 1st convolution
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                        input_shape=(input_height, input_width, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 2nd convolution
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 3rd convolution
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        # 4th convolution
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 5th convolution
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flatten the convolved images
        self.model.add(Flatten())

        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(rate=dropout_rate))
        self.model.add(Dense(16, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(l1=l1_rate, l2=l2_rate)))

        # Output Layer
        self.model.add(Dropout(rate=dropout_rate))
        self.model.add(Dense(num_outputs, activation='linear', 
                    kernel_regularizer=regularizers.L1L2(l1=l1_rate, l2=l2_rate)))

        self.model.compile(loss='mean_absolute_error', 
                        optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    

    def fit(self, x_train, y_train, epochs, batch_size, 
            x_val=None, y_val=None):
        '''
        Fit (train) the NN.
        '''
        if isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray):
            callback = EarlyStopping(monitor='val_loss', patience=10)
            self.history = self.model.fit(x_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks = [callback],
                                        validation_data = (x_val,y_val))

        else:
            callback = EarlyStopping(monitor='loss', patience=10)
            self.history = self.model.fit(x_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks = [callback])



    def evaluate(self, x_test, y_test):
        '''
        Evaluate the Model on the test set.
        '''
        self.loss, accuracy = self.model.evaluate(x_test, y_test)
        print('Evaluation Loss: ', self.loss)


    def summary(self):
        '''
        Print the summary of the NN
        '''
        self.model.summary()
