from model import PoseNet
import plots as p


def main():

    shapes = ['pose_dodec_so']
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    
    scale = False
    val = True
    outlier_bool = False
    sensor = 'relative'

    CNN = PoseNet(  option = sensor,
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

    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = CNN.read_data(shapes, path,
                                                        option = sensor,
                                                        val = val, 
                                                        combine_shapes = True,
                                                        scale_data = scale,
                                                        outlier_bool = outlier_bool,
                                                        binarise=True)

    p.plot_angle_distribution(y_train, y_test, sensor, y_val=y_val)

    CNN.create_network(x_train[0].shape[0], 240, y_train.shape[1]) # create the NN
    #CNN.create_network_di(135, 240, y_train.shape[1])
    CNN.summary()
    CNN.fit(x_train, y_train, epochs=500, batch_size=8, x_val=x_val, y_val=y_val) # train the NN
    CNN.evaluate(x_test, y_test) # evaluate the NN
    CNN.save_network(shapes, outlier_bool, scale)
    CNN.plot_learning_curves()

    loss = CNN.loss

    return 0



if __name__ == '__main__':
    main()