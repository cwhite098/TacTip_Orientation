from msilib.schema import Class
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.callbacks import EarlyStopping
import json
import matplotlib.pyplot as plt


class PoseNet():
    def __init__(self, option, conv_activation, dropout_rate,
                    l1_rate, l2_rate):
        self.model = None
        self.history = None

        # Model Hyperparameters
        self.option = option
        self.conv_activation = conv_activation
        self.dropout_rate = dropout_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate

    def create_network(self, input_height, input_width, num_outputs):
        '''
        Create the CNN model.
        '''
        self.model = Sequential()

        # 1st convolution
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                        input_shape=(input_height, input_width, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 2nd convolution
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 3rd convolution
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        # 4th convolution
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 5th convolution
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flatten the convolved images
        self.model.add(Flatten())

        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(16, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        # Output Layer
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(num_outputs, activation='linear', 
                    kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        self.model.compile(loss='mean_absolute_error', 
                        optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    

    def fit(self, x_train, y_train, epochs, batch_size, 
            x_val=None, y_val=None):
        '''
        Fit (train) the NN.
        '''
        # Save details on the training data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs

        # Train the network
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
        # Save details on the testing set
        self.x_test = x_test
        self.y_test = y_test

        # Evaluate the network
        self.loss, accuracy = self.model.evaluate(x_test, y_test)
        print('Evaluation Loss: ', self.loss)


    def summary(self):
        '''
        Print the summary of the NN
        '''
        self.model.summary()

    def plot_learning_curves(self):
        '''
        Plot the loss and the validation loss
        '''
        plt.plot(self.history.history['loss'], label='Loss')
        plt.plot(self.history.history['val_loss'], label= 'Val Loss')
        plt.legend(), plt.title('Loss Curve'), plt.show()

    def save_network(self):
        '''
        Save the network and a JSON of parameters.
        '''
        param_dict = {
            'sensors used': self.option,
            'training examples': self.x_train.shape[0],
            'eval examples': self.x_val.shape[0],
            'test examples': self.x_test.shape[0],
            'epochs': self.epochs,
            'conv activation': self.conv_activation,
            'dropout rate': self.dropout_rate,
            'l1 rate': self.l1_rate,
            'l2 rate': self.l2_rate
        }
        if self.loss:
            param_dict['loss'] = self.loss

        self.model.save('saved_nets/'+self.option+'/CNN.h5')
        with open('saved_nets/'+self.option+'/params.json', 'w') as fp:
            json.dump(param_dict, fp)
            fp.close()



