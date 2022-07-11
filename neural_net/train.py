from model import PoseNet
import plots as p
import os
import shutil
import cv2 as cv
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt



def train_test_val_split(X, Y):

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

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



def read_data(shapes, path, option, val, combine_shapes=False, scale_data=False, outlier_bool = True):
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



def main():

    shapes = ['pose_cube']
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    
    scale = False
    val = True
    outlier_bool = True
    sensor = 'object'

    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = read_data(shapes, path,
                                                        option = sensor,
                                                        val = val, 
                                                        combine_shapes = True,
                                                        scale_data = scale,
                                                        outlier_bool = outlier_bool)

    p.plot_angle_distribution(y_train, y_test, sensor, y_val=y_val)

    CNN = PoseNet(  option = sensor,
                    conv_activation = 'elu',
                    dropout_rate = 0.001,
                    l1_rate = 0.001,
                    l2_rate = 0.064,
                    learning_rate = 0.0001,
                    decay_rate = 0.000001,
                    dense_width = 16,
                    loss_func = 'mae',
                    batch_bool = True,
                    N_convs = 4,
                    N_filters = 512
                     )

    CNN.create_network(x_train[0].shape[0], 240, y_train.shape[1]) # create the NN
    CNN.summary()
    CNN.fit(x_train, y_train, epochs=500, batch_size=16, x_val=x_val, y_val=y_val) # train the NN
    CNN.evaluate(x_test, y_test) # evaluate the NN
    CNN.save_network(shapes, outlier_bool, scale)
    CNN.plot_learning_curves()

    loss = CNN.loss
    if scale:
        loss_unscaled = test_scaler.inverse_transform(np.array([loss,loss,loss,loss]))
        print(loss_unscaled)

    return 0



if __name__ == '__main__':
    main()