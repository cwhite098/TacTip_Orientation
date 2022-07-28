import argparse
import imutils
import cv2 as cv
import sys
import numpy as np
from markers.camera_calibration.calibrate_camera import calibrate
from markers.generate_markers import ARUCO_DICT


def draw_over_markers(frame, ids, corners, cam_mat, dist_mat):

    # flatten the ArUco IDs list
    ids = ids.flatten()

    # len(ids) should be 1 when there is only 1 marker
    rvec_dict = {}
    tvec_dict = {}
    for i in range(0,len(corners)):
        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.023, cam_mat, dist_mat)
        rvec_dict[str(ids[i])]=rvec.flatten()
        tvec_dict[str(ids[i])]=tvec.flatten().tolist()
        cv.aruco.drawAxis(frame, cam_mat, dist_mat, rvec, tvec, 0.01)  # Draw Axis

    # loop over the detected ArUCo corners and ids
    # and draw the bounding box and add a label
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned
        # in top-left, top-right, bottom-right, and bottom-left
        # order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
        cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
        cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

        # compute and draw the center (x, y)-coordinates of the
        # ArUco marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
        # draw the ArUco marker ID on the frame
        cv.putText(frame, str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)


    return frame, rvec_dict, tvec_dict


def find_markers(frame, **args):
    '''
    Just a cut down version of main to be imported into collect data
    Fix the type of the marker.
    I want the rvec, tvec and the frame as output
    '''
    cam_mat = args['cam_mat']
    dist_mat = args['dist_mat']

    tag_type = 'DICT_4X4_50'
    arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[tag_type])
    arucoParams = cv.aruco.DetectorParameters_create()

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame,
            arucoDict, parameters=arucoParams,
            cameraMatrix=cam_mat, distCoeff=dist_mat)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # Draw box and label any detected markers
        frame, rvec_dict, tvec_dict = draw_over_markers(frame, ids, corners, cam_mat, dist_mat)
    else:
        rvec_dict, tvec_dict = None, None

    return frame, rvec_dict, tvec_dict



def main():

    # construct the argument parser and parse the arguments
    # type should be DICT_4X4_50
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
        default="DICT_ARUCO_ORIGINAL",
        help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())


    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)


    # load the ArUCo dictionary and grab the ArUCo parameters
    print("[INFO] detecting '{}' tags...".format(args["type"]))
    arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    arucoParams = cv.aruco.DetectorParameters_create()

    # Define camera parameters - camera matrix and distortion
    [ret, cam_mat, dist_mat, rvecs, tvecs] = calibrate('camera_calibration/calibration_images')
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 1000 pixels
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = imutils.resize(frame, width=1000)
        # detect ArUco markers in the input frame
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame,
            arucoDict, parameters=arucoParams,
            cameraMatrix=cam_mat, distCoeff=dist_mat)

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # Draw box and label any detected markers
            frame, rvec_dict, tvec_dict = draw_over_markers(frame, ids, corners, cam_mat, dist_mat)
            if rejected:
                frame, rvec_dict, tvec_dict = draw_over_markers(frame, np.array([0]*len(rejected)), rejected, cam_mat, dist_mat)
        else:
            rvec_dict, tvec_dict = None, None

        # show the output frame
        cv.imshow("Frame", frame)

        print(rvec_dict)

        key = cv.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()