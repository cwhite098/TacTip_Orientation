'''
3 camera threads
Collect camera images to one frame - add text to frame with the real and the predicted angles
Angles for sensors and for the object

Replace collection thread with a prediction thread (or two, one for each NN)
Use a global frame buffer
Time the NN prediction to make sure this is feasible

Initialisation - load the networks and the camera feeds

'''
import cv2 as cv
import threading
import keyboard
import time
import os
import numpy as np
from markers.camera_calibration.calibrate_camera import calibrate
from markers.detect_markers import find_markers
from collect_data import process_sensor_frame, camera_thread
import settings
from neural_net.model import PoseNet


def binarise(frame):
    th, frame = cv.threshold(frame, 100, 1, cv.THRESH_BINARY)
    return frame

class predict_thread(threading.Thread):
    def __init__(self, sensor_net_path, object_net_path, combined_path=False):
        threading.Thread.__init__(self)
        self.sensor_net_path = sensor_net_path
        self.object_net_path = object_net_path

        if isinstance(combined_path, str):
            self.net_path = combined_path
            self.combined = True
        else:
            self.combined = False

    def run(self):

        self.load_nets()
        time.sleep(5)

        while True:
            l_sensor = binarise(settings.frame_buffer['Sensor1'])
            r_sensor = binarise(settings.frame_buffer['Sensor2'])
            input_frame = np.vstack((l_sensor, r_sensor)).reshape((1, 270,240,1))
            if self.combined:
                angles = self.sensor_net.predict(input_frame)
                settings.prediction_buffer['sensor'] = angles[:4]
                settings.prediction_buffer['object'] = angles[4:]

            else:
                sensor_angles = self.sensor_net.predict(input_frame)
                object_angles = self.object_net.predict(input_frame)
                settings.prediction_buffer['sensor'] = sensor_angles
                settings.prediction_buffer['object'] = object_angles

    def load_nets(self):
        # Init 2 posenet classses with any args
        self.object_net = PoseNet(  option = 'object',
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
        self.sensor_net = PoseNet(  option = 'sensor',
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

        if self.combined:
            self.sensor_net.load_net(self.net_path)
        else:
            self.object_net.load_net(self.object_net_path)
            self.sensor_net.load_net(self.sensor_net_path)
        


class display_thread(threading.Thread):
    def __init__(self, combined):
        threading.Thread.__init__(self)
        self.combined = combined

    def run(self):
        self.make_frame()

    def make_frame(self):
        print('[INFO] Starting, Please Wait...')
        time.sleep(10)
        while True:
            l_sensor = settings.frame_buffer['Sensor1']
            r_sensor = settings.frame_buffer['Sensor2']
            external = settings.frame_buffer['external']
            external = cv.pyrDown(cv.pyrDown(external))

            l_sensor = cv.cvtColor(l_sensor,cv.COLOR_GRAY2BGR)
            r_sensor = cv.cvtColor(r_sensor,cv.COLOR_GRAY2BGR)

            sensors = np.hstack((l_sensor, r_sensor))
            sensors_ext = np.vstack((sensors, external))

            blank = np.zeros((115,480,3), np.uint8)

            full = np.vstack((sensors_ext, blank))

            decimals = 2

            # Get the true rotations from the markers
            left_rvec = np.around(settings.rvec_buffer['left'][1:], decimals)
            right_rvec = np.around(settings.rvec_buffer['right'][1:], decimals)
            object_rvec = np.around(settings.rvec_buffer['object'], decimals)

            # Get the predicted rotations from the NNs
            try:

                if self.combined:
                    preds = np.around(settings.prediction_buffer['sensor'], decimals)
                    left_str = 'Left Sensor: '+str(left_rvec) + ' ' + str(preds[0,:2])
                    right_str = 'Right Sensor: '+str(right_rvec) + ' ' + str(preds[0,2:4])
                    object_str = str(object_rvec) + ' '+ str(preds[0,4:])

                else:
                    sensor_prediction = np.around(settings.prediction_buffer['sensor'], decimals)
                    object_prediction = np.around(settings.prediction_buffer['object'], decimals)

                    left_str = 'Left Sensor: '+str(left_rvec) + ' ' + str(sensor_prediction[0,:2])
                    right_str = 'Right Sensor: '+str(right_rvec) + ' ' + str(sensor_prediction[0,2:])
                    object_str = str(object_rvec) + ' '+ str(object_prediction)

                # Display the rotations here
                font                   = cv.FONT_HERSHEY_SIMPLEX
                fontScale              = 0.6
                fontColor              = (255,255,255)
                thickness              = 1
                lineType               = 2

                for i, st in enumerate([left_str, right_str, object_str]):
                    cv.putText(full, st, 
                        (10, 450+(i*25)), 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            except KeyError:
                print('No predictions!!')

            cv.imshow('Realtime Prediction', full)

            key = cv.waitKey(20)
            if key == 27:
                print('[INFO] Closing thread')
                break


class kill_switch(threading.Thread):
    '''
    Thread that runs in the background and will kill all threads when the
    right key is pressed.
    '''
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # The main 'loop' of the thread that looks for the button presses
        print('[INFO] Starting collection thread')
        print('Press s to save snapshot')
        keyboard.add_hotkey('x', self.exit_key)
        keyboard.wait()
    
    def exit_key(self):
        # Button press that exits the program
        keyboard.press('x')
        print('[INFO] Closing')
        os._exit(1)



def main():

    # Initialise globals
    settings.init()

    # Calibrate external camera
    print('[INFO] Calibrating camera')
    [ret, cam_mat, dist_mat, rvecs, tvecs] = calibrate('markers/camera_calibration/calibration_images')

    mode = 'so'

    # Initialise camera threads
    sensor1 = camera_thread('Sensor1', 3, (1920,1080), mode, process_sensor_frame, False, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (47, -4))
    sensor2 = camera_thread('Sensor2', 1, (1920,1080), mode, process_sensor_frame, False, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (59, -6))
    external = camera_thread('external', 2, (1920,1080), mode, find_markers, False, cam_mat=cam_mat, dist_mat=dist_mat)

    sensor_path = 'combined_Thu_Jul_21_11-22-40_2022'
    object_path = 'object_Sat_Jul_16_18-10-46_2022'
    combined_path = 'neural_net/saved_nets/'+'sensor+object_Fri_Jul_22_14-33-23_2022'
    predict = predict_thread('neural_net/saved_nets/'+sensor_path, 'neural_net/saved_nets/'+object_path, combined_path=combined_path)
    display = display_thread(True)
    x_switch = kill_switch()

    sensor1.start()
    sensor2.start()
    external.start()
    predict.start()
    display.start()
    x_switch.start()

    return 0



if __name__ =='__main__':
    main()
