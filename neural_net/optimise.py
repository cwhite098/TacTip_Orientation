from model import PoseNet
import talos as ta
import numpy as np


def train_func(x_train, y_train, x_val, y_val, params):

    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'

    CNN = PoseNet(  option = params['option'],
                    conv_activation = params['conv_activation'],
                    dropout_rate = params['dropout_rate'],
                    l1_rate = params['l1_rate'],
                    l2_rate = params['l2_rate'],
                    learning_rate = params['learning_rate'],
                    decay_rate = params['decay_rate'],
                    dense_width = params['dense_width'],
                    loss_func = params['loss_func'],
                    batch_bool = params['batch_bool'],
                    N_convs = params['N_convs'],
                    N_filters = params['N_filters'],
                    )

    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = CNN.read_data(params['shapes'], path,
                                                        option = params['option'],
                                                        val = False, 
                                                        combine_shapes = True,
                                                        scale_data = params['scale'],
                                                        outlier_bool = params['outliers'])

    CNN.create_network(x_train[0].shape[0], 240, y_train.shape[1]) # create the NN
    CNN.fit(x_train, y_train, epochs=500, batch_size = params['batch_size'], x_val=x_test, y_val=y_test) # train the NN
    #CNN.save_network(['pose_cube'], True, False)
    history, model = CNN.return_history()

    return history, model


def main():

    sensor = 'object'

    param_dict = {
            'option': ['combined'],
            'conv_activation': ['elu', 'relu'],
            'dropout_rate': [0.0001],
            'l1_rate': [0.01, 0.001, 0.0001],
            'l2_rate': [0.001, 0.01],
            'learning_rate': [0.0001, 0.00001],
            'decay_rate': [0.000001],
            'dense_width': [8, 16],
            'loss_func': ['mae', 'mse'],
            'batch_bool': [True],
            'N_convs': [4,5],
            'N_filters': [128, 256],
            'batch_size': [16],
            'scale': [False],
            'outliers': [True],
            'shapes': [['cube', 'cylinder', 'pose_dodec_so']]
    }

    x = np.array([[1],[1],[1]])
    y = np.array([1,1,1])

    t = ta.Scan(x=x,
               y=y,
               model=train_func,
               params=param_dict,
               experiment_name='sensor_pose',
               round_limit=150,
               disable_progress_bar=False)


    return 0


if __name__ == '__main__':
    main()