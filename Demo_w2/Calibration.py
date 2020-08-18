import numpy as np
import cv2
import glob


def calibrate(dirpath='calibration_data', prefix='snapshot_', image_format='jpg', square_size=2.5, width=8, height=6,
              test_accuracy=True, test_accuracy_delay=1300):
    """
    :param dirpath: path to the calibration images
    :param prefix: prefix before the image number
    :param image_format: format of the calibration images, OpenCV supports .jpg and .png
    :param square_size: size of an edge of a single chessboard square in cm
    :param width: number of intersections points in the bigger size of a chessboard
    :param height: number of intersection points in the smaller size of a chessboard
    :param test_accuracy: if True shows the intersections found during the calibration
    :param test_accuracy_delay: delay between showing the test pictures
    :return: ret, camera calibration matrix, camera distortion matrix, rotation vectors, transformation vectors
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath + '/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Display the corners
            if test_accuracy:
                img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                cv2.imshow('Accuracy test', img)
                cv2.waitKey(test_accuracy_delay)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def save_coefficients(matrix, distortion, path='calibration_results/'):
    np.savetxt(path + 'camera_matrix.txt', matrix)  # Save camera matrix
    np.savetxt(path + 'camera_distortion.txt', distortion)  # Save camera distortion


def make_calibration_images():
    """Press j to make a snapshot, q to close the window"""
    cap = cv2.VideoCapture(0)
    num = 28
    while (key := cv2.waitKey(1) & 0xFF) != ord('q'):
        _, frame = cap.read()
        cv2.imshow('Screenshot frame', cv2.flip(frame, 1))

        if key == ord('j'):
            cv2.imwrite(f'calibration_data/snapshot_{(num := num + 1)}.jpg', frame)


if __name__ == '__main__':
    # make_calibration_images()
    _, mtx, dist, *_ = calibrate(test_accuracy=False)
    save_coefficients(mtx, dist)
