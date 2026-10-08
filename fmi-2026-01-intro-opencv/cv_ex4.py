import cv2

if __name__ == '__main__':
    # cap = cv2.VideoCapture('data/test.mp4')
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('video')
    cv2.createTrackbar('minThreshold', 'video', 40, 255, lambda x: None)
    cv2.createTrackbar('maxThreshold', 'video', 150, 255, lambda x: None)
    if cap.isOpened():
        while(cap.isOpened() and cv2.waitKey(30) != ord('q')):
            ret, frame = cap.read()
            if not ret:
                print('video ends')
                break
            cv2.imshow('frame', frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            minThreshold = cv2.getTrackbarPos('minThreshold', 'video')
            maxThreshold = cv2.getTrackbarPos('maxThreshold', 'video')
            edges = cv2.Canny(frame_gray, minThreshold, maxThreshold)
            # print(edges.shape)

            cv2.imshow('edges', edges)
    else:
        print('video not opened')

    cap.release()
    cv2.destroyAllWindows()