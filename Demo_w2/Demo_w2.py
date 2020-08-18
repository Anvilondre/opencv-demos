import numpy as np
import cv2
import cv2.aruco as aruco

marker_size = 9

camera_matrix = np.loadtxt('calibration_results/camera_matrix.txt')
camera_distortion = np.loadtxt('calibration_results/camera_distortion.txt')

aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dictionary, parameters=parameters)

    if ids is not None and len(ids) > 0:
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

    cv2.imshow('frame', cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
