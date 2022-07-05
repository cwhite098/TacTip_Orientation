import matplotlib.pyplot as plt
import numpy as np


def plot_angle_distribution(y_train, y_test, sensor, y_val=None):


    if sensor == 'combined':

        plt.subplot(1,2,1)
        plt.scatter(y_train[:,0], y_train[:,1], label='Train')
        plt.scatter(y_test[:,0], y_test[:,1], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,1], label='Val')
        plt.legend(), plt.xlabel('Psi1'), plt.ylabel('Phi1')
        plt.title('Sensor 1 Angles')

        plt.subplot(1,2,2)
        plt.scatter(y_train[:,2], y_train[:,3], label='Train')
        plt.scatter(y_test[:,2], y_test[:,3], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,2], y_val[:,3], label='Val')
        plt.legend(), plt.xlabel('Psi2'), plt.ylabel('Phi2')
        plt.title('Sensor 2 Angles')

    else:
        plt.scatter(y_train[:,0], y_train[:,1], label='Train')
        plt.scatter(y_test[:,0], y_test[:,1], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,1], label='Val')
        plt.legend(), plt.xlabel('Psi'), plt.ylabel('Phi')
        plt.title(sensor + ' Sensor Angles')

    plt.show()


    return 0