import cv2
import numpy as np
import ctypes
import os
import time

# ---------------------------------------------------------------------------
# 1. MANUALLY BIND THE OPENNI2 C-DLL INTERFACE
# ---------------------------------------------------------------------------
# Paste the path to your extracted Orbbec OpenNI SDK 'Redist' folder here:
OPENNI_REDIST_PATH = r"D:\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"

os.add_dll_directory(OPENNI_REDIST_PATH)
oni_dll = ctypes.cdll.LoadLibrary(os.path.join(OPENNI_REDIST_PATH, "OpenNI2.dll"))


# Define OpenNI2 C-API Enums and Structs
class OniFrame(ctypes.Structure):
    _fields_ = [
        ("dataSize", ctypes.c_int), ("data", ctypes.c_void_p),
        ("sensorType", ctypes.c_int), ("timestamp", ctypes.c_uint64),
        ("frameIndex", ctypes.c_int), ("width", ctypes.c_int),
        ("height", ctypes.c_int), ("videoMode", ctypes.c_int),
        ("croppingEnabled", ctypes.c_int), ("cropOriginX", ctypes.c_int),
        ("cropOriginY", ctypes.c_int), ("stride", ctypes.c_int),
        ("pixelFormat", ctypes.c_int)
    ]


# Initialize OpenNI2 C-Core Engine
if oni_dll.oniInitialize(2) != 0:
    print("[ERROR] Failed to initialize native OpenNI2 DLL context.")
    exit()
print("[SUCCESS] C-DLL Engine Handshake complete.")

# Open Depth Device Endpoints
device_handle = ctypes.c_void_p()
if oni_dll.oniDeviceOpen(None, ctypes.byref(device_handle)) != 0:
    print("[ERROR] No Orbbec camera detected by OpenNI2 drivers.")
    oni_dll.oniShutdown()
    exit()

# Create and Start Raw Depth Stream
stream_handle = ctypes.c_void_p()
oni_dll.oniDeviceCreateStream(device_handle, 1, ctypes.byref(stream_handle))  # 1 = ONI_SENSOR_DEPTH
oni_dll.oniStreamStart(stream_handle)
print("[SUCCESS] Hardware Depth Laser fired.")

# ---------------------------------------------------------------------------
# 2. BIND THE STANDARD WINDOWS UVC COLOR WEBCAM (Index 0 or 1 from your scan)
# ---------------------------------------------------------------------------
color_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStreaming initialized cleanly. Press 'q' to shut down.")

try:
    while True:
        # --- Read OpenNI Depth Frame Buffer ---
        frame_ptr = ctypes.c_void_p()
        # Read frame with a quick timeout check
        if oni_dll.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr)) == 0:
            oni_frame = OniFrame.from_address(frame_ptr.value)

            # Extract raw 16-bit short pointer array from the DLL buffer
            data_size = oni_frame.width * oni_frame.height
            BufferType = ctypes.c_uint16 * data_size
            raw_buffer = BufferType.from_address(oni_frame.data)

            # Map raw buffer space cleanly to a 2D NumPy array
            depth_array = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(480, 640)

            # Normalize 16-bit integers to a visual 8-bit scale
            depth_scaled = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            cv2.imshow("Astra Pro - Native Depth (Direct DLL)", depth_colored)

            # Safe internal pointer release inside the C-driver tier
            oni_dll.oniFrameRelease(frame_ptr)

        # --- Read Generic UVC Webcam RGB Frame ---
        ret, color_frame = color_cap.read()
        if ret:
            cv2.imshow("Astra Pro - Native RGB (DirectShow)", color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Clean Teardown ---
    print("\nStopping native hardware endpoints...")
    oni_dll.oniStreamStop(stream_handle)
    oni_dll.oniStreamDestroy(stream_handle)
    oni_dll.oniDeviceClose(device_handle)
    oni_dll.oniShutdown()
    color_cap.release()
    cv2.destroyAllWindows()
    print("Clean shutdown successful.")
