import matplotlib.pyplot as plt
import numpy as np


def plot_angle_distribution(y_train, y_test):

    plt.subplot(1,2,1)
    plt.scatter(y_train[:,0], y_train[:,1], label='Train')
    plt.scatter(y_test[:,0], y_test[:,1], label='Test')
    plt.legend(), plt.xlabel('Psi1'), plt.ylabel('Phi1')
    plt.title('Sensor 1 Angles')

    plt.subplot(1,2,2)
    plt.scatter(y_train[:,2], y_train[:,3], label='Train')
    plt.scatter(y_test[:,2], y_test[:,3], label='Test')
    plt.legend(), plt.xlabel('Psi2'), plt.ylabel('Phi2')
    plt.title('Sensor 2 Angles')

    plt.show()


    return 0