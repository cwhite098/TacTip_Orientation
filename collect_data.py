import cv2 as cv
import threading
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
        print('Starting thread '+self.display_name)
        if self.display_bool:
            print('Displaying '+self.display_name)
        else:
            print('Not displaying '+self.display_name)
        display_camera(self.display_name, self.camera_id, self.resolution, self.processing_func, display_bool=self.display_bool, **self.process_args)


def display_camera(display_name, camera_id, resolution, processing_func, display_bool=True, **process_args):

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

        if display_bool:
            if type(frame) == tuple:
                cv.imshow(display_name, frame[0])
            else:
                cv.imshow(display_name, frame)

        key = cv.waitKey(20)
        if key == 27:
            print('Closing thread: ' + display_name)
            break
    cv.destroyWindow(display_name)



def test_camera(camera_id):
    cap = cv.VideoCapture(camera_id) 
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', camera_id)


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
    print('Calibrating camera...')
    [ret, cam_mat, dist_mat, rvecs, tvecs] = calibrate('markers/camera_calibration/calibration_images')

    # Test camera sources
    test_camera(3)
    test_camera(2)
    test_camera(1)

    # Initialise camera threads
    sensor1 = camera_thread('Sensor1', 3, (1920,1080), process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (47, -4))
    sensor2 = camera_thread('Sensor2', 1, (1920,1080), process_sensor_frame, True, res=(240,135), crop = [300, 0, 1600, 1080], threshold = (59, -6))
    external = camera_thread('external', 2, (480,270), find_markers, True, cam_mat=cam_mat, dist_mat=dist_mat)

    sensor1.start()
    sensor2.start()
    external.start()


if __name__ == '__main__':
    main()
