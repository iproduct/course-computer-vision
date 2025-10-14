import cv2
import numpy as np

if __name__ == "__main__":
    # read a color image
    # cap = cv2.VideoCapture('test.mp4')
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while(cv2.waitKey(30) != ord('q')):
            ret, frame = cap.read()
            if ret == False:
                print('Video ends')
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Canny method for edges detection
            # argument: image_array, minVal, maxVal
            # above maxVal: sure-edges, below minVal : non-edges
            # between maxVal and minVal : depend on the connectivity to sure-edges
            edges = cv2.Canny(frame, 70, 250)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # vis = np.concatenate((frame_gray, edges), axis=1)
            alpha = 0.5
            beta = (1.0 - alpha);
            vis = cv2.addWeighted(frame, alpha, edges_bgr, beta, 0.0);
            cv2.imshow('video', vis)
    else:
        print('Video opening failed')

    cap.release()
    cv2.destroyAllWindows()