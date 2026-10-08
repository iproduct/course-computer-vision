import cv2

if __name__ == '__main__':
    # cap = cv2.VideoCapture('data/test.mp4')
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while(cap.isOpened() and cv2.waitKey(30) != ord('q')):
            ret, frame = cap.read()
            if not ret:
                print('video ends')
                break
            cv2.imshow('frame', frame)
    else:
        print('video not opened')

    cap.release()
    cv2.destroyAllWindows()