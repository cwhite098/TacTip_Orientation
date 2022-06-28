from importlib.resources import path
from pickle import READONLY_BUFFER
import cv2 as cv
import threading
from threading import Event
import keyboard
import time
import json
import os
import sys
import numpy as np
from multiprocessing import Process
from markers.camera_calibration.calibrate_camera import calibrate
from markers.detect_markers import find_markers

class camera_thread(threading.Thread):
    def __init__(self, display_name, camera_id, resolution, processing_func=None, display_bool=True, **process_args):
        threading.Thread.__init__(self)
        self.display_name = display_name
        self.camera_id = camera_id
        self.resolution = resolution
        self.display_bool = display_bool
        self.processing_func = processing_func
        self.process_args = process_args

    def run(self):
        print('[INFO] Starting thread '+self.display_name)
        if self.display_bool:
            print('[INFO] Displaying '+self.display_name)
        else:
            print('[INFO] Not displaying '+self.display_name)
        display_camera(self.display_name, self.camera_id, self.resolution, self.processing_func, display_bool=self.display_bool, **self.process_args)


class collection_thread(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.path = path

    def run(self):
        # The main 'loop' of the thread that looks for the button presses
        print('[INFO] Starting collection thread')
        print('Press s to save snapshot')
        keyboard.add_hotkey('s', self.save_key)
        keyboard.add_hotkey('x', self.exit_key)
        keyboard.wait()
    
    def exit_key(self):
        # Button press that exits the program
        keyboard.press('x')
        print('[INFO] Closing Data Collection')
        os._exit(1)

    def save_key(self):
        # Button press that saves a snap shot of the system
        keyboard.press('s')
        print('[INFO] Saving')
        global stamp
        stamp = str(time.ctime())
        stamp=stamp.replace(' ', '_')
        stamp=stamp.replace(':', '-')
        os.mkdir(PATH+stamp)
        capture.set()
        global capture_counter
        time.sleep(3)
        print('Press s to save snapshot')

        if capture_counter >= 3:
            capture.clear()
            capture_counter = 0 
                

def process_angles(rvec_dict):
    '''
    Function that processes the raw angle data.
    Makes the sensor rotations relative to some other marker
    '''
    try:
        reference1 = rvec_dict['1']
        reference2 = rvec_dict['2']
        l_sensor = rvec_dict['4']
        r_sensor = rvec_dict['3']

        # Get the average of the 2 reference markers
        reference = (reference1+reference2)/2

        # If the 2 reference markers vary dramatically, there is an error, return 0s
        if (np.abs(reference1 - reference2) > 0.5).any():
            print('[INFO] Bad Reference Pose')
            dict = {'left': 0, 'right': 0, 'reference': 0}
            return dict

        # If the sensors have moved too far in the last 5 frames,
        # set the rotations to the averages of those frames
        if (np.abs(l_sensor - np.mean(l_buffer,axis=0)) > 0.75).any() or  (np.abs(r_sensor - np.mean(r_buffer,axis=0)) > 0.75).any():
            print('[INFO] Bad Sensor Pose')
            l_sensor = np.mean(l_buffer,axis=0)
            r_sensor = np.mean(r_buffer,axis=0)
        else:
            l_buffer.append(l_sensor)
            r_buffer.append(r_sensor)

            if len(l_buffer)>5:
                l_buffer.pop(0)
            if len(r_buffer)>5:
                r_buffer.pop(0)

        l_vec = np.mean(l_buffer,axis=0) - reference # get the relative rotations
        r_vec = np.mean(r_buffer,axis=0) - reference # use the average to get more reliable data

        # Ensure the rotations are in [-pi, pi]
        for i in range(len(r_vec)):
            if np.abs(r_vec[i]) > 3.1416:
                r_vec[i] = r_vec[i]-np.sign(r_vec[i])*2*np.pi
        for i in range(len(l_vec)):
            if np.abs(l_vec[i]) > 3.1416:
                l_vec[i] = l_vec[i]-np.sign(l_vec[i])*2*np.pi

        # Convert to list for json dump
        l_vec = l_vec.tolist()
        r_vec = r_vec.tolist()
        reference = reference.tolist()

        dict = {'left': l_vec, 'right': r_vec, 'reference': reference}
        print(dict['left'], dict['right'])

    except KeyError:
        print('[INFO] Marker not found, returning raw rotations')
        dict = {'left': 0, 'right': 0, 'reference': 0}

    except TypeError:
        print('[INFO] Marker Obscured, returning raw data')
        dict = {'left': 0, 'right': 0, 'reference': 0}

    return dict


def display_camera(display_name, camera_id, resolution, processing_func, display_bool=True, **process_args):

    cv.namedWindow(display_name)

    global capture_counter
    global stamp
    global reference_buffer
    reference_buffer = []
    global r_buffer
    r_buffer = []
    global l_buffer
    l_buffer = []

    cam = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    #cam.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0]) # set horizontal res
    #cam.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1]) # set vertical res
    cam.set(3, resolution[0])
    cam.set(4, resolution[1])

    if cam.isOpened(): # this is where the external camera is breaking
        ret, frame = cam.read()
    else:
        ret=False

    while ret:

        ret, frame = cam.read()

        if processing_func is not None:
            frame = processing_func(frame, **process_args)

        if type(frame) == tuple:
            tvec_dict = frame[2]
            rvec_dict = frame[1]
            rvec_dict = process_angles(rvec_dict)
            #print(rvec_dict['left'])
            frame = frame[0]
        else:
            tvec_dict, rvec_dict = False, False


        if display_bool:
            cv.imshow(display_name, frame)

        # if data saving is occurring
        if capture.is_set():
            # write the image to the folder
            cv.imwrite(PATH+ stamp + '/' + display_name + '.png', frame)
            if rvec_dict:
                # dump the rotation data
                with open(PATH+stamp+'/rvec.json', 'w') as fp:
                    json.dump(rvec_dict, fp)
                    fp.close()
                with open(PATH+stamp+'/tvec.json', 'w') as fp:
                    json.dump(tvec_dict, fp)
                    fp.close()

            capture_counter += 1
            time.sleep(3)

        key = cv.waitKey(20)
        if key == 27:
            print('[INFO] Closing thread: ' + display_name)
            break
    cv.destroyWindow(display_name)



def process_sensor_frame(frame, **args):

    crop = args['crop']
    threshold = args['threshold']
    resolution = args['res']
    
    # Crop the sensor videos
    if crop is not None:
        x0, y0, x1, y1 = crop
        frame = frame[y0:y1, x0:x1]

    width, offset = threshold
 	
    # Convert to grayscale
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Apply Gaussian filter
    frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, width, offset) # apply Gaussian adaptive threshold
    # Change to specified size
    frame = cv.resize(frame, resolution, interpolation=cv.INTER_AREA)

    return frame


def main():

    # Calibrate external camera
    print('[INFO] Calibrating camera')
    [ret, cam_mat, dist_mat, rvecs, tvecs] = calibrate('markers/camera_calibration/calibration_images')

    # Init vars for capturing snapshots
    global capture
    capture = Event()
    global capture_counter
    capture_counter = 0

    object = input('Enter the name of the object being used:')
    global PATH
    PATH = 'data/'+object+'/'

    # Initialise camera threads
    sensor1 = camera_thread('Sensor2', 3, (1920,1080), process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (47, -4))
    sensor2 = camera_thread('Sensor1', 1, (1920,1080), process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (59, -6))
    external = camera_thread('external', 2, (1920,1080), find_markers, True, cam_mat=cam_mat, dist_mat=dist_mat)

    capture_thread = collection_thread('data/'+object+'/')

    sensor1.start()
    sensor2.start()
    external.start()
    capture_thread.start()


if __name__ == '__main__':
    main()
