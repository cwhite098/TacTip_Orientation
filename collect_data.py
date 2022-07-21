from distutils.log import error
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
import settings

class camera_thread(threading.Thread):
    def __init__(self, display_name, camera_id, resolution, mode, processing_func=None, display_bool=True, **process_args):
        threading.Thread.__init__(self)
        self.display_name = display_name
        self.camera_id = camera_id
        self.resolution = resolution
        self.display_bool = display_bool
        self.processing_func = processing_func
        self.process_args = process_args
        self.mode = mode

    def run(self):
        print('[INFO] Starting thread '+self.display_name)
        if self.display_bool:
            print('[INFO] Displaying '+self.display_name)
        else:
            print('[INFO] Not displaying '+self.display_name)
        self.display_camera(self.display_name, self.camera_id, self.resolution, self.processing_func, display_bool=self.display_bool, **self.process_args)



    def display_camera(self, display_name, camera_id, resolution, processing_func, display_bool=True, **process_args):

        if display_bool:
            cv.namedWindow(display_name)

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
                rvec_dict = self.process_angles(rvec_dict)
                settings.rvec_buffer = rvec_dict
                #print(rvec_dict['left'])

                frame = frame[0]
            else:
                tvec_dict, rvec_dict = False, False


            if display_bool:
                cv.imshow(display_name, frame)

            # if data saving is occurring
            if settings.capture.is_set():
                # write the image to the folder
                cv.imwrite(PATH+ settings.stamp + '/' + display_name + '.png', frame)
                if rvec_dict:
                    # dump the rotation data
                    with open(PATH+settings.stamp+'/rvec.json', 'w') as fp:
                        json.dump(rvec_dict, fp)
                        fp.close()
                    with open(PATH+settings.stamp+'/tvec.json', 'w') as fp:
                        json.dump(tvec_dict, fp)
                        fp.close()

                settings.capture_counter += 1
                time.sleep(3)

            if settings.error.is_set():
                # Collect some readings to find the error in the pose
                # estimation
                if rvec_dict:
                    settings.error_buffer.append([rvec_dict['left'][1:], rvec_dict['right'][1:]])
            
            settings.frame_buffer[display_name] = frame

            key = cv.waitKey(20)
            if key == 27:
                print('[INFO] Closing thread: ' + display_name)
                break
        cv.destroyWindow(display_name)

        


    def process_angles(self, rvec_dict):
        '''
        Function that processes the raw angle data.
        Makes the sensor rotations relative to some other marker
        '''
        buffer_size = 10

        if self.mode == 's':
            try: # get markers for sensor tracking mode
                reference1 = rvec_dict['1']
                reference2 = rvec_dict['2']
                l_sensor = rvec_dict['4']
                r_sensor = rvec_dict['3']

            except KeyError:
                print('[INFO] Marker not found, returning raw rotations')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict
            except TypeError:
                print('[INFO] Marker Obscured, returning raw data')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict
            
            dict = self.get_relative_rotations(  [reference1, reference2], [l_sensor, r_sensor], ['left', 'right'],
                                            [settings.l_buffer, settings.r_buffer], buffer_size)

            
        elif self.mode == 'o':
            try: # get markers for object tracking mode
                reference1 = rvec_dict['1']
                reference2 = rvec_dict['2']
                object1 = rvec_dict['6']

            except KeyError:
                print('[INFO] Marker not found, returning raw rotations')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict
            except TypeError:
                print('[INFO] Marker Obscured, returning raw data')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict

            dict = self.get_relative_rotations(  [reference1, reference2], [object1], ['object'],
                                            [settings.object_buffer], buffer_size)

        elif self.mode  == 'so':
            try: # get markers for object tracking mode
                reference1 = rvec_dict['1']
                reference2 = rvec_dict['2']
                l_sensor = rvec_dict['4']
                r_sensor = rvec_dict['3']
                object1 = rvec_dict['6']

            except KeyError:
                print('[INFO] Marker not found, returning raw rotations')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict
            except TypeError:
                print('[INFO] Marker Obscured, returning raw data')
                dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
                return dict

            dict = self.get_relative_rotations(  [reference1, reference2], [object1, l_sensor, r_sensor], ['object', 'left', 'right'],
                                            [settings.object_buffer, settings.l_buffer, settings.r_buffer], buffer_size)


        print(dict)
        return dict


    def get_relative_rotations(self, refs, markers, names, buffers, buffer_size):

        output_rots = []

        # Get the average of the 2 reference markers
        reference = (refs[0]+refs[1])/2

        # If the 2 reference markers vary dramatically, there is an error, return 0s
        if (np.abs(refs[0] - refs[1]) > 0.5).any():
            print('[INFO] Bad Reference Pose')
            dict = {'object': [0,0], 'left': [0,0], 'right': [0,0], 'reference': [0,0]}
            return dict

        # If the sensors have moved too far in the last 5 frames,
        # set the rotations to the averages of those frames
        for i in range(len(markers)):
            if (np.abs(markers[i] - np.mean(buffers[i],axis=0)) > 0.75).any():
                print('[INFO] Bad Sensor Pose')
                markers[i] = np.mean(buffers[i], axis=0)
            else:
                buffers[i].append(markers[i])
                # Check buffer size and remove entry if necessary
                if len(buffers[i])>buffer_size:
                    buffers[i].pop(0)

            output_rots.append(np.mean(buffers[i],axis=0) - reference) # get the relative rotations
            
        # Ensure the rotations are in [-pi, pi]
        for i, output in enumerate(output_rots):
            if np.abs(output[i]) > 3.1416:
                output[i] = output[i]-np.sign(output[i])*2*np.pi
            # convert to list for json dump
            output_rots[i] = output.tolist()
        
        # Convert reference
        reference = reference.tolist()

        # Collect outputs into dictionary
        dict = {}
        for i, output in enumerate(output_rots):
            dict[names[i]] = output
        dict['reference'] = reference

        # Missing the printing of the angles during data collection

        return dict


    
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
        keyboard.add_hotkey('e', self.pose_error)
        keyboard.wait()
    
    def exit_key(self):
        # Button press that exits the program
        keyboard.press('x')
        print('[INFO] Closing Data Collection')
        os._exit(1)

    def pose_error(self):
        keyboard.press('e')
        print('[INFO] Collecting measurements for precision analysis')
        settings.error.set()
        time.sleep(10)
        error_array = np.array(settings.error_buffer)
        psi1 = np.ptp(error_array[:,0,0])
        phi1 = np.ptp(error_array[:,0,1])
        psi2 = np.ptp(error_array[:,1,0])
        phi2 = np.ptp(error_array[:,1,1])
        print('Ranges: \n', 
                'Psi1: ', psi1,'\n',
                'Phi1: ', phi1,'\n',
                'Psi2: ', psi2,'\n',
                'Phi2: ', phi2)
        time.sleep(3)
        settings.error_buffer = []
        settings.error.clear()


    def save_key(self):
        # Button press that saves a snap shot of the system
        keyboard.press('s')
        print('[INFO] Saving')
        settings.stamp = str(time.ctime())
        settings.stamp=settings.stamp.replace(' ', '_')
        settings.stamp=settings.stamp.replace(':', '-')
        os.mkdir(PATH+settings.stamp)
        settings.capture.set()
        time.sleep(3)
        print('Press s to save snapshot')

        if settings.capture_counter >= 3:
            settings.capture.clear()
            settings.capture_counter = 0 
                

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

    # initialise globals
    settings.init()

    # Calibrate external camera
    print('[INFO] Calibrating camera')
    [ret, cam_mat, dist_mat, rvecs, tvecs] = calibrate('markers/camera_calibration/calibration_images')


    object = input('Enter the name of the object being used:')
    mode = input('Collecting data for sensor pose (s) or object pose (o):')
    global PATH
    PATH = 'data/'+object+'/'

    # Initialise camera threads
    sensor1 = camera_thread('Sensor1', 3, (1920,1080), mode, process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (47, -4))
    sensor2 = camera_thread('Sensor2', 1, (1920,1080), mode, process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (59, -6))
    external = camera_thread('external', 2, (1920,1080), mode, find_markers, True, cam_mat=cam_mat, dist_mat=dist_mat)

    capture_thread = collection_thread('data/'+object+'/')

    sensor1.start()
    sensor2.start()
    external.start()
    capture_thread.start()


if __name__ == '__main__':
    main()
