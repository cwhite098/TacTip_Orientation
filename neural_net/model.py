import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import time
import os
import cv2 as cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shutil

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPUs found: ', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def train_test_val_split(X, Y):

    train_ratio = 0.75
    validation_ratio = 0.05
    test_ratio = 0.2

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    return x_train, x_test, x_val, y_train, y_test, y_val


def remove_outliers(X, Y, lower, upper):
    '''
    Function to remove outliers in the angles
    '''
    print('[INFO] Removing Outliers')
    Y = np.array(Y)
    X = np.array(X)
    upper_percentile = np.percentile(Y, upper, axis=0)
    lower_percentile = np.percentile(Y, lower, axis=0)

    print('[INFO] Upper Percentile: ', upper_percentile)
    print('[INFO] Lower Percentile: ', lower_percentile)

    # use np.argwhere to find the data that is to be kept
    to_keep_idx = np.argwhere((Y<upper_percentile) & (Y>lower_percentile))

    idx = np.unique(to_keep_idx[:,0])

    X = X[idx]
    Y = Y[idx]

    print('[INFO] Remaining exmples: ', X.shape[0])

    return X, Y


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

        stamp = str(time.ctime())
        stamp=stamp.replace(' ', '_')
        self.stamp=stamp.replace(':', '-')


    def read_data(self, shapes, path, option, val, combine_shapes=False, scale_data=False, outlier_bool = True):
        '''
        Function that reads the data that has been collected for the different shapes
        Carries out a train/test split to be used to train NN.

        option : str
            Can be 'left' or 'right' for individual sensors.
            Can be 'dual' for one net with both sensors being input.
            Can be 'combined' for stacked images -> 4 angles.
        '''

        X = []
        Y = []

        available_shapes = os.listdir(path)
        for s in shapes:
            snapshots = os.listdir(path+'/'+s)
            for example in snapshots:

                # Read the images
                sensor1 = cv.imread(path+'/'+s+'/'+example+'/sensor1.png', cv.IMREAD_GRAYSCALE)
                sensor2 = cv.imread(path+'/'+s+'/'+example+'/sensor2.png', cv.IMREAD_GRAYSCALE)

                # Read the orientation data
                with open(path+'/'+s+'/'+example+'/rvec.json', 'r') as f:
                    rvec_dict = json.load(f)

                if option == 'object':
                    try:
                        object_rot = rvec_dict['object']
                    except KeyError:
                        print('Zero Rotation Found: '+example)
                        #shutil.rmtree(path+'/'+s+'/'+example)
                        object_rot = 0
                    # Check if the data is good and if so, save.
                    if isinstance(object_rot, int):
                        print('ZERO ROTATION FOUND: '+example)
                        shutil.rmtree(path+'/'+s+'/'+example)
                    else:
                        X.append(np.vstack((sensor1, sensor2)))
                        target = np.array(object_rot)
                        Y.append(target)
                
                else:
                    sensor1_rot = rvec_dict['left']
                    sensor2_rot = rvec_dict['right']
                    # Check if the data is good and if so, save.
                    if isinstance(sensor1_rot, int) & isinstance(sensor2_rot, int):
                        print('ZERO ROTATION FOUND: '+example)
                    else:
                        if option == 'combined':
                            X.append(np.vstack((sensor1, sensor2)))
                            target = np.array([sensor1_rot[1], sensor1_rot[2], sensor2_rot[1], sensor2_rot[2]])
                            Y.append(target)
                        elif option == 'left':
                            X.append(sensor1)
                            target = np.array([sensor1_rot[1], sensor1_rot[2]])
                            Y.append(target)
                        elif option == 'right':
                            X.append(sensor2)
                            target = np.array([sensor2_rot[1], sensor2_rot[2]])
                            Y.append(target)
                        elif option == 'dual':
                            X.append(sensor1)
                            X.append(sensor2)
                            target = np.array([sensor1_rot[1], sensor1_rot[2]])
                            Y.append(target)
                            target = np.array([sensor2_rot[1], sensor2_rot[2]])
                            Y.append(target)
                    del sensor1, sensor2


        # Function to remove outliers
        if outlier_bool:
            X, Y = remove_outliers(X, Y, 20, 80)

        # train/test split
        if not val:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
            y_val = None
            x_val = None
            del X, Y
        else:
            x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(X, Y)

        # Scale the angles - is this necessary???
        train_scaler = StandardScaler()
        test_scaler = StandardScaler()
        val_scaler = StandardScaler()
        if scale_data:
            y_train_scaled = train_scaler.fit_transform(y_train)
            y_test_scaled = test_scaler.fit_transform(y_test)
            if y_val:
                y_val_scaled = val_scaler.fit_transform(y_val)
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test
            y_val_scaled = y_val

        # Print info
        print('Training Examples: ', len(x_train))
        print('Training Labels: ', len(y_train))
        print('Testing Examples: ', len(x_test))
        print('Testing Labels: ', len(y_test))
        if val:
            print('Validation Examples: ', len(x_val))
            print('Validation Labels: ', len(y_val))
        print('Size of image: ', x_train[1].shape)
        print('Shape of Targets: ', y_test[0].shape)

        # Reshape for insertion into network
        x_train = np.array(x_train).reshape(len(x_train), x_train[0].shape[0],240,1)
        x_test = np.array(x_test).reshape(len(x_test), x_train[0].shape[0],240,1)
        if val:
            x_val = np.array(x_val).reshape(len(x_val), x_train[0].shape[0],240,1)
            return x_train, x_test, x_val, np.array(y_train_scaled), np.array(y_test_scaled), np.array(y_val_scaled), test_scaler
        else:
            return x_train, x_test, None, np.array(y_train_scaled), np.array(y_test_scaled), None, test_scaler


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
            self.model.add(Conv2D(filters=self.N_filters, kernel_size=(3,3), padding='same'))
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
        plt.savefig('saved_nets/'+self.option+'_'+self.stamp+'/learning-curve.pdf')
        plt.legend(), plt.title('Loss Curve'), plt.show()


    def return_history(self):
        return self.model.history, self.model


    def save_network(self, shapes, outliers_removed, data_scaled):
        '''
        Save the network and a JSON of parameters.
        '''
        param_dict = {
            'sensors used': self.option,
            'training examples': self.x_train.shape[0],
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

        self.model.save('saved_nets/'+self.option+'_'+self.stamp+'/CNN.h5')
        with open('saved_nets/'+self.option+'_'+self.stamp+'/params.json', 'w') as fp:
            json.dump(param_dict, fp)
            fp.close()


    def load_net(self, path):

        print('[INFO] Loading Model')
        self.model = load_model(path+'/CNN.h5')

    def predict(self, input):

        angles = self.model.predict(input)
        return angles


