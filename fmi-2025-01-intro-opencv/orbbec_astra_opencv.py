import cv2
import numpy as np

# 1. Initialize Depth Stream via OpenCV's Built-in Astra Backend
# CAP_OPENNI2_ASTRA targets the private depth protocol over USB
print("Connecting to Astra Depth Engine via OpenCV Native Drivers...")
depth_cap = cv2.VideoCapture(cv2.CAP_OPENNI2_ASTRA)

# 2. Initialize Color Stream via standard Windows Webcam index
# Try 0, 1, or 2 if your laptop has a built-in webcam
print("Connecting to Astra RGB UVC Camera...")
color_cap = cv2.VideoCapture(1)

if not depth_cap.isOpened():
    print("\n[ERROR] OpenCV could not access the Depth Sensor.")
    print("Ensure you installed the Orbbec Sensor Driver .exe from the SDK package.")
    exit()

print("\nStreams online! Press 'q' to exit.")

while True:
    # --- Capture Depth Maps ---
    # In OpenCV OpenNI2 mode, grab() reads the frame buffer
    if depth_cap.grab():
        # Retrieve the raw 16-bit depth map matrix
        ret_d, depth_map = depth_cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
        if ret_d:
            # Convert 16-bit millimeters into an 8-bit visible grayscale range
            depth_scaled = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            cv2.imshow("Native OpenCV - Depth Map", depth_colored)

    # --- Capture RGB Frames ---
    ret_c, color_frame = color_cap.read()
    if ret_c:
        cv2.imshow("Native OpenCV - RGB Webcam", color_frame)

    # Exit cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Teardown
depth_cap.release()
color_cap.release()
cv2.destroyAllWindows()
print("Streams released.")