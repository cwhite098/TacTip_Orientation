import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(photo_path, image_format='jpg', square_size=0.025, width=9, height=6):
    '''
    Apply the calibration given a set of 20 images
    '''
    objp = np.zeros((height*width,3), np.float32) # this is the chessboard matrix
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

    objp = objp*square_size

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    images = os.listdir(photo_path)

    for file in images:

        img = cv.imread(photo_path+'/'+file) # read the photo from the file
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
        
        # find the corners of the chessboard
        ret, corners = cv.findChessboardCorners(gray, (width,height), None)

        if ret:
            # If found, add the real world and image points
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw the corners and display
            img = cv.drawChessboardCorners(img, (width,height), corners2, ret)
            #cv.imshow('Chessboard',img)
            #cv.waitKey(0)

    # Calibrate the camera using the collected points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_parameters(mtx, dists, path):
    '''
    Save the camera matrix and the distortion coefficients.
    '''
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dists)

    cv_file.release()

def load_parameters(path):
    '''
    Load the camera matrix and the distortion coefficients
    '''
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    cam_mat = cv_file.getNode('K').mat()
    dists = cv_file.getNode('D').mat()

    cv_file.release()

    return cam_mat, dists



def main():
    [ret, mtx, dist, rvecs, tvecs] = calibrate('calibration_images')
    save_parameters(mtx, dist, 'parameters')


if __name__ == '__main__':
    main()

