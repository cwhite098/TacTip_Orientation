import os
import json
import matplotlib.pyplot as plt
import numpy as np
import model
import matplotlib


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 13}

matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=12)


def plot_angle_distribution(y_train, y_test, sensor, y_val=None):

    y_train = rad_to_degree(y_train)
    y_test = rad_to_degree(y_test)


    if sensor == 'combined':

        plt.subplot(1,2,1)
        plt.scatter(y_train[:,0], y_train[:,1], label='Train')
        plt.scatter(y_test[:,0], y_test[:,1], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,1], label='Val')
        plt.legend(), plt.xlabel('Yaw'), plt.ylabel('Pitch')
        plt.title('Sensor 1 Angles')

        plt.subplot(1,2,2)
        plt.scatter(y_train[:,2], y_train[:,3], label='Train')
        plt.scatter(y_test[:,2], y_test[:,3], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,2], y_val[:,3], label='Val')
        plt.legend(), plt.xlabel('Yaw'), plt.ylabel('Pitch')
        plt.title('Sensor 2 Angles')

    elif sensor == 'object':
        plt.subplot(1,2,1)
        plt.scatter(y_train[:,0], y_train[:,1], label='Train')
        plt.scatter(y_test[:,0], y_test[:,1], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,1], label='Val')
        plt.legend(), plt.xlabel('Object Roll'), plt.ylabel('Object Yaw')
        plt.title('Object Pitch vs Roll')

        plt.subplot(1,2,2)
        plt.scatter(y_train[:,0], y_train[:,2], label='Train')
        plt.scatter(y_test[:,0], y_test[:,2], label='Test')
        if isinstance(y_val, np.ndarray):
            plt.scatter(y_val[:,0], y_val[:,2], label='Val')
        plt.legend(), plt.xlabel('Object Roll'), plt.ylabel('Object Pitch')
        plt.title('Object Yaw vs Roll')


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

    true = (true/(2*np.pi))*360
    pred = (pred/(2*np.pi))*360

    for i in range(no_angles):
        plt.subplot(1,no_angles,i+1)
        plt.tight_layout()
        
        plt.plot([np.min(true), np.max(true)],[np.min(true), np.max(true)], c='r', linewidth=2, zorder=1)
        plt.scatter(true[:,i], pred[:,i], marker='*', s=10, zorder=2)
        plt.xlabel('True Angles '+str(i)), plt.ylabel('Predicted Angle '+str(i))
        plt.grid()
    plt.show()

def rad_to_degree(array):
    array = (array/(2*np.pi))*360
    return array

def plot_error_distributions():

    '''
    Reads the angles from the 'error' folder and plots them as histograms to show the variation
    that occurs when using the aruco markers for pose estimation.
    '''
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data/error/'
    snapshots = os.listdir(path)
    rots = ['Roll', 'Pitch', 'Yaw']
    left_list = []
    right_list = []
    object_list = []
    ref_list = []
    for example in snapshots:
        # Read the orientation data
        with open(path+example+'/rvec.json', 'r') as f:
            rvec_dict = json.load(f)
            left_list.append(rvec_dict['left'])
            right_list.append(rvec_dict['right'])
            object_list.append(rvec_dict['object'])
            ref_list.append(rvec_dict['reference'])
    left_list = rad_to_degree(np.array(left_list))
    right_list = rad_to_degree(np.array(right_list))
    object_list = rad_to_degree(np.array(object_list))
    ref_list = rad_to_degree(np.array(ref_list))

    # Add some code to get means and std of each of the angles
    # Display this information in the titles of the plots
    list_of_lists = [left_list, right_list, object_list, ref_list]
    stats = []
    for i in range(4):
        list = list_of_lists[i]
        for i in range(3):
            mean = np.around(np.mean(list[:,i]),3)
            std = np.around(np.std(list[:,i]),3)
            stats.append([mean, std])

    bin_no = 20

    for i in range(3):
        plt.subplot(4,3,i+1)
        plt.hist(left_list[:,i], bins=bin_no, edgecolor='black', linewidth=1.2)
        plt.title('Left '+rots[i]+' '+str(stats[i]))
    for i in range(3):
        plt.subplot(4,3,i+4)
        plt.hist(right_list[:,i], bins=bin_no, edgecolor='black', linewidth=1.2)
        plt.title('Right '+rots[i]+' '+str(stats[i+3]))
    for i in range(3):
        plt.subplot(4,3,i+7)
        plt.hist(object_list[:,i], bins=bin_no, edgecolor='black', linewidth=1.2)
        plt.title('Object '+rots[i]+' '+str(stats[i+6]))
    for i in range(3):
        plt.subplot(4,3,i+10)
        plt.hist(ref_list[:,i], bins=bin_no, edgecolor='black', linewidth=1.2)
        plt.title('Reference '+rots[i]+' '+str(stats[i+9]))

    plt.tight_layout()
    plt.show()

    return 0


def main():


    plot_error_distributions()
    
    net_path = 'saved_nets/object_Mon_Jul_25_17-05-24_2022'

    sensor_net = model.PoseNet(  option = 'object',
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
    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = sensor_net.read_data(['pose_dodec_so', 'pose_dodec'], path,
                                                        option = 'object',
                                                        val = True, 
                                                        combine_shapes = True,
                                                        scale_data = False,
                                                        outlier_bool = False,
                                                        binarise=True)

    #plot_angle_distribution(y_train, y_test, 'combined', y_val=None)

    pred = sensor_net.predict(x_test)

    plot_prediction_line(y_test, pred, 3)
    
    return 0


if __name__ == '__main__':
    main()