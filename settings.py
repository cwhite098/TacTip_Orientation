'''
File for initialising global variables
'''
from threading import Event


def init():

    global reference_buffer
    reference_buffer = []

    global r_buffer
    r_buffer = []
    
    global l_buffer
    l_buffer = []
    
    global error_buffer
    error_buffer = []
    
    global object_buffer
    object_buffer = []

    global rvec_buffer
    rvec_buffer = {}

    global frame_buffer
    frame_buffer = {}

    global capture
    capture = Event()

    global capture_counter
    capture_counter = 0

    global error
    error = Event()

    global stamp
    stamp = ''



