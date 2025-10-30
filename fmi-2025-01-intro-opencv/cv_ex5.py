import cv2
import numpy as np

def nothing(x):
    pass

if __name__ == "__main__":
    # read a color image
    # cap = cv2.VideoCapture('test.mp4')

    cv2.namedWindow("video")
    # cv2.createTrackbar("Min", "video", 40, 255, nothing)
    # cv2.createTrackbar("Max", "video", 255, 255, nothing)

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        while cv2.waitKey(30) != ord('q'):
            ret, frame = cap.read()
            if not ret:
                print('Video ends')
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # identity_kernel = np.array([[0,0,0], [0,1,0], [0,0,0]])
            # identity_img = cv2.filter2D(src = frame_gray, ddepth=-1, kernel=identity_kernel)
            gausian_blur = cv2.GaussianBlur(frame_gray, (11, 11), sigmaX = 16, sigmaY = 16)
            cv2.imshow('video', gausian_blur)
    else:
        print('Video opening failed')

    cap.release()
    cv2.destroyAllWindows()