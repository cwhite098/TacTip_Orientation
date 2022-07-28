import cv2 as cv
import threading
import keyboard
import time
import json
import os
import numpy as np
from markers.camera_calibration.calibrate_camera import calibrate
from markers.detect_markers import find_markers
import settings

class camera_thread(threading.Thread):
    '''
    Class that creates a new thread to capture the stream from an attached camera.
    '''
    def __init__(self, display_name, camera_id, resolution, mode, path, processing_func=None, display_bool=True, **process_args):
        '''
        Parameters
        ----------
        display_name : str
            String that forms the name of the window displaying the camera.
        camera_id : int
            The number used to identify the camera for the string - from your OS.
        resolution : tuple
            A tuple containing the horizontal and vertical resolution to capture from the camera.
        mode : str
            The data collection mode, either 's' for sensor, 'o' for object or 'so' for sensor and object.
            This corresponds to the poses which will be tracked.
        processing_func : function
            A function that performs some processing on each frame. Must have frame (array) as input and output.
        display_bool : bool
            True displays the camera feed, False does not.
        **process_args
            Optional arguments that will be passed to the processing function.
        '''
        threading.Thread.__init__(self)
        self.display_name = display_name # name of the display window
        self.camera_id = camera_id
        self.resolution = resolution
        self.display_bool = display_bool
        self.processing_func = processing_func # a function to apply processing to each frame
        self.process_args = process_args
        self.mode = mode # the data collection mode
        self.path = path

    def run(self):
        '''
        Function that runs when the thread is initialised.
        '''
        print('[INFO] Starting thread '+self.display_name)
        if self.display_bool:
            print('[INFO] Displaying '+self.display_name)
        else:
            print('[INFO] Not displaying '+self.display_name)
        self.display_camera(self.display_name, self.camera_id, self.resolution, self.processing_func, display_bool=self.display_bool, **self.process_args)



    def display_camera(self, display_name, camera_id, resolution, processing_func, display_bool=True, **process_args):
        '''
        Function that reads the camera, applies processing and displays the frames
        Parameters
        ----------
        display_name : str
            String that forms the name of the window displaying the camera.
        camera_id : int
            The number used to identify the camera for the string - from your OS.
        resolution : tuple
            A tuple containing the horizontal and vertical resolution to capture from the camera.
        processing_func : function
            A function that performs some processing on each frame. Must have frame (array) as input and output.
        display_bool : bool
            True displays the camera feed, False does not.
        **process_args
            Optional arguments that will be passed to the processing function.
        '''

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
            # Apply the provided processing
            if processing_func is not None:
                frame = processing_func(frame, **process_args)

            # If the processing func returns multiple vars they
            # need to be unpacked.
            if type(frame) == tuple:
                tvec_dict = frame[2]
                rvec_dict = frame[1]
                rvec_dict = self.verif_angles(rvec_dict)
                settings.rvec_buffer = rvec_dict # add the rvec to the global buffer
                frame = frame[0]
            else:
                tvec_dict, rvec_dict = False, False

            if display_bool:
                cv.imshow(display_name, frame)

            # if data saving is occurring
            if settings.capture.is_set():
                # write the image to the folder
                cv.imwrite(self.path+ settings.stamp + '/' + display_name + '.png', frame)
                if rvec_dict:
                    # dump the rotation data
                    with open(self.path+settings.stamp+'/rvec.json', 'w') as fp:
                        json.dump(rvec_dict, fp)
                        fp.close()
                    with open(self.path+settings.stamp+'/tvec.json', 'w') as fp:
                        json.dump(tvec_dict, fp)
                        fp.close()
                # Increment the global capture counter so the programme knows when all 3
                # frames have been collected
                settings.capture_counter += 1
                time.sleep(3)
            
            settings.frame_buffer[display_name] = frame

            key = cv.waitKey(20)
            if key == 27:
                print('[INFO] Closing thread: ' + display_name)
                break
        cv.destroyWindow(display_name)

        

    def verif_angles(self, rvec_dict):
        '''
        Function that verifies that all the required markers are detected, and if so
        they angles are sent to be processed.

        Parameters
        ----------
        rvec_dict : dict
            Dictionary containing the marker numbers and their associated poses.

        Returns
        -------
        dict : dict
            Dictionary containing the poses of all the objects/sensors being tracked.
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
            
            dict = self.process_rotations(  [reference1, reference2], [l_sensor, r_sensor], ['left', 'right'],
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

            dict = self.process_rotations(  [reference1, reference2], [object1], ['object'],
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

            dict = self.process_rotations(  [reference1, reference2], [object1, l_sensor, r_sensor], ['object', 'left', 'right'],
                                            [settings.object_buffer, settings.l_buffer, settings.r_buffer], buffer_size)

        print(dict)
        return dict


    def process_rotations(self, refs, markers, names, buffers, buffer_size):
        '''
        Function that processes the angles when they have been found from the markers.

        Parameters
        ----------
        refs : list
            List containing the poses of the two reference markers
        markers : list
            List containing the poses of the object/sensors.
        names : list
            List containing strings for the names of each sensor/object to be used when 
            saving the data as a json.
        buffers : list
            List containing the global buffers that are used to hold and verify
            the pose data.
        buffer_size : int
            The maximum length of the buffers containing the pose data.

        Returns
        -------
        dict : dict
            Dictionary containing the poses of all the objects/sensors being tracked.
        '''

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

            output_rots.append(np.mean(buffers[i],axis=0)) # get the rotations from the buffer
            
        # Ensure the rotations are in [-pi, pi]
        for i, output in enumerate(output_rots):
            if np.abs(output[i]) > np.pi:
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

        return dict


    
class collection_thread(threading.Thread):
    '''
    Thread that facilitates the operator interacting with the data collection process.
    '''
    def __init__(self, path):
        '''
        Parameters
        ----------
        path : str
            The path to the data folder in the workspace.
        '''
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

    def save_key(self):
        # Button press that saves a snap shot of the system
        keyboard.press('s')
        print('[INFO] Saving')
        settings.stamp = str(time.ctime())
        settings.stamp=settings.stamp.replace(' ', '_')
        settings.stamp=settings.stamp.replace(':', '-')
        os.mkdir(self.path+settings.stamp)
        settings.capture.set()
        time.sleep(3)
        print('Press s to save snapshot')

        if settings.capture_counter >= 3:
            settings.capture.clear()
            settings.capture_counter = 0 
                

def process_sensor_frame(frame, **args):
    '''
    Function that processes the frames captured by the tactips.
    
    Parameters
    ----------
    **args
        Passed to the camera thread as optional arguments:
        crop : tuple
            The desired coordinates to crop the frame at. [x0,y0,x1,y1]
        threshold : tuple
            The parameters for the Gaussian adaptive thresholding: (width, offset)
        resolution : tuple
            The resolution to return the frame as (used in resizing).
    
    Returns
    ------
    frame : np.array
        An array containing the data for the processed frame.
    '''
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
    mode = input('Collecting data for sensor pose (s), object pose (o), or both (so):')

    path = 'data/'+object+'/'

    # Initialise camera threads
    sensor1 = camera_thread('Sensor1', 3, (1920,1080), mode, path, process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (47, -4))
    sensor2 = camera_thread('Sensor2', 1, (1920,1080), mode, path, process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (59, -6))
    external = camera_thread('external', 2, (1920,1080), mode, path, find_markers, True, cam_mat=cam_mat, dist_mat=dist_mat)

    # Initialise the capture thread
    capture_thread = collection_thread(path)

    # Begin the collection
    sensor1.start()
    sensor2.start()
    external.start()
    capture_thread.start()


if __name__ == '__main__':
    main()
