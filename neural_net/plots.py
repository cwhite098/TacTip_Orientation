import matplotlib.pyplot as plt
import numpy as np
import train


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

    elif sensor == 'object':
        plt.subplot(1,2,1)
        plt.scatter(y_train[:,0], y_train[:,1], label='Train')
        plt.scatter(y_test[:,0], y_test[:,1], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,1], label='Val')
        plt.legend(), plt.xlabel('F/B'), plt.ylabel('S/S')
        plt.title('Sensor 1 Angles')

        plt.subplot(1,2,2)
        plt.scatter(y_train[:,0], y_train[:,2], label='Train')
        plt.scatter(y_test[:,0], y_test[:,2], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,2], label='Val')
        plt.legend(), plt.xlabel('F/B'), plt.ylabel('U/D')
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

def plot_angle_hists(y_train, y_test, y_val=None):

    if isinstance(y_val, np.ndarray):
        plt.subplot(1,3,1)
        plt.hist(y_train[:,0])
        plt.title('Angle 1')

        plt.subplot(1,3,2)
        plt.hist(y_train[:,1])
        plt.title('Angle 2')

        plt.subplot(1,3,3)
        plt.hist(y_train[:,2])
        plt.title('Angle 3')

    plt.show()



def main():
    shapes = ['cube','cylinder']
    shapes = ['pose_cube']
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    
    scale = False
    val = True
    sensor = 'object'

    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = train.read_data(shapes, path,
                                                        option = sensor,
                                                        val = val, 
                                                        combine_shapes = True,
                                                        scale_data = scale)

    plot_angle_hists(y_train, y_test, y_val=y_val)

    return 0


if __name__ == '__main__':
    main()