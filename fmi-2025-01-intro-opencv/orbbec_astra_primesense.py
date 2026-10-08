import cv2
import numpy as np

import sys
from collections.abc import Mapping

# Create a temporary fallback class to satisfy the legacy metaclass instantiation
class MockAbstractClass:
    pass

# Force inject it into the type system to intercept the "abstract class" check
import abc
abc.ABC = MockAbstractClass

# Import the fixed primesense openni2 wrapper instead
from primesense import openni2

# 1. Initialize OpenNI2
# Pass your absolute path to the Orbbec SDK Redist folder if auto-seeking fails
try:
    openni2.initialize()
    print("OpenNI2 framework initialized successfully via primesense.")
except Exception as e:
    print(f"Initialization failure: {e}")
    print("Tip: If it fails, pass the path string: openni2.initialize('C:/YourPath/OpenNI2/Redist')")
    exit()

# 2. Connect to the Orbbec Depth Endpoints
try:
    device = openni2.Device.open_any()
    depth_stream = device.create_depth_stream()
    depth_stream.start()
    print("Depth stream opened successfully.")
except Exception as e:
    print(f"Failed to find or open depth sensor hardware: {e}")
    openni2.unload()
    exit()

# 3. Connect to the UVC Webcam component (Astra Pro RGB Sensor)
# If your PC has a built-in webcam, change 0 to 1 or 2
color_cap = cv2.VideoCapture(0)

print("\nStreaming active. Click an image window and press 'q' to close.")

while True:
    # --- Process OpenNI Depth Stream ---
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()

    # Read raw 16-bit short integers and reshape into depth map grid matrix
    depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(480, 640)

    # Normalize array into a standard 8-bit visual image mapping
    depth_scaled = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

    # --- Process UVC RGB Stream ---
    ret, color_frame = color_cap.read()

    # --- Draw Live Previews ---
    cv2.imshow("Orbbec Astra Pro - Depth (OpenNI)", depth_colored)
    if ret:
        cv2.imshow("Orbbec Astra Pro - RGB (UVC)", color_frame)

    # Break loop on 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safe Environment Teardown
depth_stream.stop()
openni2.unload()
color_cap.release()
cv2.destroyAllWindows()
print("System closed cleanly.")