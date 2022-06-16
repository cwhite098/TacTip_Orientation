import cv2 as cv
import threading
from multiprocessing import Process

class camera_thread(threading.Thread):
    def __init__(self, display_name, camera_id, resolution, display_bool=True):
        threading.Thread.__init__(self)
        self.display_name = display_name
        self.camera_id = camera_id
        self.resolution = resolution
        self.display_bool = display_bool

    def run(self):
        print('Starting thread '+self.display_name)
        if self.display_bool:
            print('Displaying '+self.display_name)
        else:
            print('Not displaying '+self.display_name)
        display_camera(self.display_name, self.camera_id, self.resolution, display_bool=self.display_bool)


def display_camera(display_name, camera_id, resolution, display_bool=True):

    cv.namedWindow(display_name)

    cam = cv.VideoCapture(camera_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0]) # set horizontal res
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1]) # set vertical res

    if cam.isOpened():
        ret, frame = cam.read()
    else:
        ret=False

    while ret:

        if display_bool:
            cv.imshow(display_name, frame)

        ret, frame = cam.read()
        key = cv.waitKey(20)
        if key == 27:
            print('Closing thread: ' + display_name)
            break
    cv.destroyWindow(display_name)


def main():
    sensor1 = camera_thread('Sensor1', 0, (240,135), True)
    sensor2 = camera_thread('Sensor2', 1, (240,135), False)

    sensor1.start()
    sensor2.start()


if __name__ == '__main__':
    main()
