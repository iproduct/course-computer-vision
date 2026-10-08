import cv2
import numpy as np
import ctypes
import os

# ===========================================================================
# 1. SETUP HARDWARE PATHS & DEFINE NATIVE STRUCTURES
# ===========================================================================

OPENNI_REDIST_PATH = r"D:\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"

os.add_dll_directory(OPENNI_REDIST_PATH)
oni_dll = ctypes.cdll.LoadLibrary(os.path.join(OPENNI_REDIST_PATH, "OpenNI2.dll"))


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


# OpenNI2 Native Hardware Property IDs
STREAM_PROPERTY_MIRRORING = 3
STREAM_PROPERTY_AUTO_EXPOSURE = 23
STREAM_PROPERTY_EXPOSURE = 22
STREAM_PROPERTY_GAIN = 24


def set_stream_mirroring(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_MIRRORING, ctypes.byref(value), ctypes.sizeof(value))


def set_ir_auto_exposure(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.byref(value),
                                 ctypes.sizeof(value))


def set_ir_exposure(stream_handle, value_int: int):
    set_ir_auto_exposure(stream_handle, False)
    val = ctypes.c_int(value_int)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_EXPOSURE, ctypes.byref(val), ctypes.sizeof(val))


def set_ir_gain(stream_handle, value_int: int):
    val = ctypes.c_int(value_int)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_GAIN, ctypes.byref(val), ctypes.sizeof(val))


# ===========================================================================
# 2. INITIALIZE HARDWARE ENDPOINTS
# ===========================================================================

if oni_dll.oniInitialize(2) != 0:
    print("[ERROR] Failed to initialize native OpenNI2 DLL context.")
    exit()

device_handle = ctypes.c_void_p()
if oni_dll.oniDeviceOpen(None, ctypes.byref(device_handle)) != 0:
    print("[ERROR] No Orbbec camera detected by OpenNI2 drivers.")
    oni_dll.oniShutdown()
    exit()

stream_handle = ctypes.c_void_p()
oni_dll.oniDeviceCreateStream(device_handle, 1, ctypes.byref(stream_handle))
oni_dll.oniStreamStart(stream_handle)

set_stream_mirroring(stream_handle, False)

# Set initial default hardware baselines
set_ir_exposure(stream_handle, 200)
set_ir_gain(stream_handle, 100)

color_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
# 3. SETUP TRACKBAR CONTROLS (FULL MANUAL MODE)
# ===========================================================================
WINDOW_NAME = "Astra Pro - Master Control Panel"
cv2.namedWindow(WINDOW_NAME)


# Trackbar event callback fallbacks
def on_exposure_change(val):
    set_ir_exposure(stream_handle, max(1, val))


def on_gain_change(val):
    set_ir_gain(stream_handle, max(1, val))


def nothing(val):
    pass


# --- Hardware Sliders ---
cv2.createTrackbar("IR Exposure", WINDOW_NAME, 200, 1000, on_exposure_change)
cv2.createTrackbar("IR Gain", WINDOW_NAME, 100, 500, on_gain_change)

# --- Software Alignment Alignment Sliders ---
# Offset sliders allow negative adjustments by mapping values relative to an offset value
cv2.createTrackbar("Shift X Offset", WINDOW_NAME, 50, 100, nothing)  # 50 means 0 shift (Range: -50 to +50 px)
cv2.createTrackbar("Shift Y Offset", WINDOW_NAME, 54, 100, nothing)  # 50 means 0 shift (Range: -50 to +50 px)
cv2.createTrackbar("Scale FX (x100)", WINDOW_NAME, 102, 150, nothing)  # 102 means 1.02 zoom multiplier

print("\nAll controls initialized. Adjust sliders to fine-tune alignment and depth metrics.")

# ===========================================================================
# 4. EXECUTION RUNTIME STREAMING LOOP
# ===========================================================================

try:
    while True:
        frame_ptr = ctypes.c_void_p()
        aligned_depth = np.zeros((480, 640), dtype=np.uint16)

        # Read sliders live values on every frame loop iteration
        cur_shift_x = cv2.getTrackbarPos("Shift X Offset", WINDOW_NAME) - 50
        cur_shift_y = cv2.getTrackbarPos("Shift Y Offset", WINDOW_NAME) - 50
        cur_scale_fx = cv2.getTrackbarPos("Scale FX (x100)", WINDOW_NAME) / 100.0

        if oni_dll.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr)) == 0:
            oni_frame = OniFrame.from_address(frame_ptr.value)
            data_size = oni_frame.width * oni_frame.height
            BufferType = ctypes.c_uint16 * data_size
            raw_buffer = BufferType.from_address(oni_frame.data)
            raw_depth = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(480, 640)

            # Apply your manual slider transform matrix directly
            M = np.float32([[cur_scale_fx, 0, cur_shift_x], [0, cur_scale_fx, cur_shift_y]])
            aligned_depth = cv2.warpAffine(raw_depth, M, (640, 480), flags=cv2.INTER_NEAREST)
            # aligned_depth = cv2.flip(warped_depth, 1)

            # Simple closing morphology to bridge micro reflection errors smoothly
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            aligned_depth = cv2.morphologyEx(aligned_depth, cv2.MORPH_CLOSE, kernel)

            oni_dll.oniFrameRelease(frame_ptr)

        ret, raw_color_frame = color_cap.read()

        if ret and np.any(aligned_depth):
            color_frame = cv2.flip(raw_color_frame, 1)
            target_x, target_y = 320, 240

            final_raw = aligned_depth[target_y, target_x]
            if final_raw > 0:
                distance_text = f"Distance: {final_raw / 10.0:.1f} cm"
            else:
                distance_text = "Distance: Signal Absorbed (0 mm)"

            # Draw visual telemetry indicators
            cv2.circle(color_frame, (target_x, target_y), 6, (0, 0, 255), -1)
            cv2.putText(color_frame, distance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_frame, f"Matrix Shift X: {cur_shift_x}px | Y: {cur_shift_y}px", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Generate depth visualization map
            depth_scaled = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            blended_view = cv2.addWeighted(color_frame, 0.6, depth_colored, 0.4, 0)

            # Display views
            cv2.imshow("Astra Pro - RGB Overlay", color_frame)
            cv2.imshow(WINDOW_NAME, blended_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    oni_dll.oniStreamStop(stream_handle)
    oni_dll.oniStreamDestroy(stream_handle)
    oni_dll.oniDeviceClose(device_handle)
    oni_dll.oniShutdown()
    color_cap.release()
    cv2.destroyAllWindows()
