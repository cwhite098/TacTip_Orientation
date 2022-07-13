from train import read_data
from model import PoseNet
import plots as p
import talos as ta
import numpy as np


def train_func(x_train, y_train, x_val, y_val, params):


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

    CNN.create_network(x_train[0].shape[0], 240, y_train.shape[1]) # create the NN
    CNN.fit(x_train, y_train, epochs=500, batch_size = params['batch_size'], x_val=x_val, y_val=y_val) # train the NN
    #CNN.save_network(['pose_cube'], True, False)
    history, model = CNN.return_history()

    return history, model


def main():

    shapes = ['pose_cube']
    path = 'C:/Users/chris/OneDrive/Uni/Summer_Research_Internship/Project/TacTip_Orientation/data'
    
    scale = False
    val = False
    outlier_bool = True
    sensor = 'object'

    x_train, x_test, x_val, y_train, y_test, y_val, test_scaler = read_data(shapes, path,
                                                        option = sensor,
                                                        val = val, 
                                                        combine_shapes = True,
                                                        scale_data = scale,
                                                        outlier_bool = outlier_bool)

    param_dict = {
            'option': [sensor],
            'conv_activation': ['elu', 'relu'],
            'dropout_rate': [0.01, 0.001, 0.0001],
            'l1_rate': [0.01, 0.001, 0.0001],
            'l2_rate': [0.064, 0.001],
            'learning_rate': [0.001, 0.0001],
            'decay_rate': [0.000001],
            'dense_width': [8, 16],
            'loss_func': ['mae', 'mse'],
            'batch_bool': [True, False],
            'N_convs': [2,3,4],
            'N_filters': [128, 256, 512],
            'batch_size': [8,16]
    }

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    t = ta.Scan(x=x,
               y=y,
               model=train_func,
               params=param_dict,
               experiment_name='optim1',
               round_limit=50,
               disable_progress_bar=False)


    return 0


if __name__ == '__main__':
    main()