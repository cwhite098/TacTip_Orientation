import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import time

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPUs found: ', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class PoseNet():
    def __init__(self, option, conv_activation, dropout_rate,
                    l1_rate, l2_rate, learning_rate, decay_rate, dense_width, loss_func,
                    batch_bool = True, N_convs=4, N_filters=512):
        self.model = None
        self.history = None

        # Model Hyperparameters
        self.option = option
        self.conv_activation = conv_activation
        self.dropout_rate = dropout_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.dense_width = dense_width
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.loss_func = loss_func
        self.batch_bool = batch_bool
        self.N_convs = N_convs
        self.N_filters = N_filters


    def create_network(self, input_height, input_width, num_outputs):
        '''
        Create the CNN model.
        '''
        self.model = Sequential()

        # 1st convolution
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                        input_shape=(input_height, input_width, 1)))
        if self.batch_bool:
            self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 2nd convolution
        for i in range(self.N_convs):
            self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
            if self.batch_bool:
                self.model.add(BatchNormalization())
            self.model.add(Activation(self.conv_activation))
            self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flatten the convolved images
        self.model.add(Flatten())

        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(self.dense_width, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        # Output Layer
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(num_outputs, activation='linear', 
                    kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        self.model.compile(loss=self.loss_func, 
                        optimizer=Adam(learning_rate=self.learning_rate, decay = self.decay_rate), 
                        metrics=['mae', 'mse'])
    

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
        self.loss, self.mae, self.mse = self.model.evaluate(x_test, y_test)
        print('Evaluation MAE: ', self.mae)


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


    def save_network(self, shapes, outliers_removed, data_scaled):
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
            'l2 rate': self.l2_rate,
            'Dense width': self.dense_width,            
            'learning rate': self.learning_rate,
            'decay rate': self.decay_rate,
            'loss function': self.loss_func,
            'Shapes used': shapes,
            'Outliers removed?': outliers_removed,
            'Data scaled?': data_scaled
        }

        if self.loss:
            param_dict['MAE'] = self.mae
            param_dict['loss'] = self.loss
            param_dict['MSE'] = self.mse

        stamp = str(time.ctime())
        stamp=stamp.replace(' ', '_')
        stamp=stamp.replace(':', '-')

        self.model.save('saved_nets/'+self.option+'_'+stamp+'/CNN.h5')
        with open('saved_nets/'+self.option+'_'+stamp+'/params.json', 'w') as fp:
            json.dump(param_dict, fp)
            fp.close()



