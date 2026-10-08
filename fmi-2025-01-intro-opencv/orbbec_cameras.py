import cv2
import numpy as np

print("Opening direct Windows hardware streams...")

# Initialize all three detected camera indices using DirectShow
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Force the cameras to request standard 640x480 streaming buffers
for cap in [cap0, cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Streaming raw feeds. Press 'q' to close windows.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # --- Render Window 0 ---
    if ret0:
        # Check if this stream is outputting 16-bit raw depth data
        if frame0.dtype == np.uint16 or (len(frame0.shape) == 2):
            depth_vis = cv2.normalize(frame0, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Camera Index 0 (Depth-Mode Map)", cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))
        else:
            cv2.imshow("Camera Index 0", frame0)

    # --- Render Window 1 ---
    if ret1:
        cv2.imshow("Camera Index 1", frame1)

    # --- Render Window 2 ---
    if ret2:
        cv2.imshow("Camera Index 2", frame2)

    # Break loop cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely close hardware communication channels
cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Streams released safely.")