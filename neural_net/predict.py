from model import create_network
import os
import cv2 as cv
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def read_data(shapes, path, combine_shapes=False):
    '''
    Function that reads the data that has been collected for the different shapes
    Carries out a train/test split to be used to train NN.
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
            sensor1_rot = rvec_dict['left']
            sensor2_rot = rvec_dict['right']

            # Check if the data is good and if so, save.
            if isinstance(sensor1_rot, int) & isinstance(sensor2_rot, int):
                print('ZERO ROTATION FOUND: '+example)
            else:
                target = np.array([sensor1_rot[1], sensor1_rot[2], sensor2_rot[1], sensor2_rot[2]])
                Y.append(target)
                X.append(np.vstack((sensor1, sensor2)))


    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # Scale the angles - is this necessary???
    train_scaler = StandardScaler()
    test_scaler = StandardScaler()
    y_train = train_scaler.fit_transform(y_train)
    y_test = test_scaler.fit_transform(y_test)

    # Print info
    print('Training Examples: ', len(x_train))
    print('Training Labels: ', len(y_train))
    print('Testing Examples: ', len(x_test))
    print('Training Labels: ', len(y_test))
    print('Size of image: ', x_train[1].shape)

    x_train = np.array(x_train).reshape(len(x_train), 270,240,1)
    x_test = np.array(x_test).reshape(len(x_test), 270,240,1)

    return x_train, x_test, np.array(y_train), np.array(y_test)



def main():

    shapes = ['cube']
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    x_train, x_test, y_train, y_test = read_data(shapes, path, False)

    print(x_train.shape)

    model = create_network(270, 240, 4)

    history = model.fit(x_train, y_train, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)

    plt.plot(history.history['loss'][1:])
    plt.title('Loss Curve'), plt.show()

    plt.plot(history.history['accuracy'])
    plt.title('Accuracy Curve'), plt.show()

    return 0



if __name__ == '__main__':
    main()