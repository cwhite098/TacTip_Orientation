import matplotlib.pyplot as plt
import numpy as np
import model


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


def plot_prediction_line(true, pred, no_angles):

    for i in range(no_angles):
        plt.subplot(1,no_angles,i+1)
        plt.plot([np.min(true), np.max(true)],[np.min(true), np.max(true)])
        plt.scatter(true[:,i], pred[:,i])
        plt.xlabel('True Angles '+str(i)), plt.ylabel('Predicted Angle '+str(i))
    plt.show()



def main():

    net_path = 'neural_net/saved_nets/combined_Mon_Jul_18_16-20-32_2022'

    sensor_net = model.PoseNet(  option = 'sensor',
                    conv_activation = 'elu',
                    dropout_rate = 0.001,
                    l1_rate = 0.0001,
                    l2_rate = 0.01,
                    learning_rate = 0.00001,
                    decay_rate = 0.000001,
                    dense_width = 16,
                    loss_func = 'mse',
                    batch_bool = False,
                    N_convs = 4,
                    N_filters = 512
                     )

    sensor_net.load_net(net_path)

    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = sensor_net.read_data(['pose_dodec_so', 'cube', 'cylinder'], path,
                                                        option = 'combined',
                                                        val = True, 
                                                        combine_shapes = True,
                                                        scale_data = False,
                                                        outlier_bool = True,
                                                        binarise=True)

    pred = sensor_net.predict(x_test)

    plot_prediction_line(y_test, pred, 4)

    return 0


if __name__ == '__main__':
    main()