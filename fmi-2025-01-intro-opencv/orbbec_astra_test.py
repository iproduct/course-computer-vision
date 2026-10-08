import cv2
import numpy as np
from openni import openni2

# Initialize OpenNI2 directly with your local OpenNI package
# Download OpenNI SDK if missing: https://orbbec.com
try:
    openni2.initialize() # Windows will auto-seek installed paths, or pass Redist path string
    print("OpenNI2 framework initialized successfully.")
except Exception as e:
    print(f"Initialization failure: {e}")
    exit()

# Connect directly to the underlying OpenNI depth endpoints
device = openni2.Device.open_any()
depth_stream = device.create_depth_stream()
depth_stream.start()

# Connect to the UVC webcam component
color_cap = cv2.VideoCapture(1) # Try 0, 1, or 2 based on your camera index

while True:
    # 1. Process OpenNI depth loop
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(480, 640)
    depth_scaled = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

    # 2. Process UVC RGB loop
    ret, color_frame = color_cap.read()

    cv2.imshow("Native OpenNI Depth Feed", depth_colored)
    if ret:
        cv2.imshow("UVC RGB Webcam Feed", color_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

depth_stream.stop()
openni2.unload()
color_cap.release()
cv2.destroyAllWindows()