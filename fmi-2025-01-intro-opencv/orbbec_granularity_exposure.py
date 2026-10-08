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
STREAM_PROPERTY_GAIN = 24  # Hardware amplification control ID


def set_stream_mirroring(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_MIRRORING, ctypes.byref(value), ctypes.sizeof(value))


def set_ir_auto_exposure(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.byref(value),
                                 ctypes.sizeof(value))


def set_ir_exposure(stream_handle, value_int: int):
    # Auto-exposure must be turned off for manual values to take effect
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
oni_dll.oniDeviceCreateStream(device_handle, 1, ctypes.byref(stream_handle))  # 1 = SENSOR_DEPTH
oni_dll.oniStreamStart(stream_handle)

set_stream_mirroring(stream_handle, False)

# Set initial default baselines
set_ir_exposure(stream_handle, 200)
set_ir_gain(stream_handle, 100)

color_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
# 3. SETUP TRACKBAR CONTROLS
# ===========================================================================
# Create a dedicated window layer specifically to house our calibration sliders
WINDOW_NAME = "Astra Pro - Fine Tuning Controls"
cv2.namedWindow(WINDOW_NAME)


# Trackbar event fallbacks
def on_exposure_change(val):
    # Enforce a minimum threshold value of 1 to prevent system lock
    exposure = max(1, val)
    set_ir_exposure(stream_handle, exposure)


def on_gain_change(val):
    gain = max(1, val)
    set_ir_gain(stream_handle, gain)


# Build our UI slider elements (Ranges mapped to legacy Astra firmware safety limits)
cv2.createTrackbar("IR Exposure", WINDOW_NAME, 200, 1000, on_exposure_change)
cv2.createTrackbar("IR Gain", WINDOW_NAME, 100, 500, on_gain_change)

# Alignment defaults
FOCAL_LENGTH_PIXELS = 580.0
LENS_BASELINE_MM = 75.0
SHIFT_Y_STATIC = 4
SCALE_F = 1.02

print("\nSliders loaded successfully. Move them to watch dark areas recalculate depth data.")

# ===========================================================================
# 4. EXECUTION RUNTIME STREAMING LOOP
# ===========================================================================

try:
    while True:
        frame_ptr = ctypes.c_void_p()
        aligned_depth = np.zeros((480, 640), dtype=np.uint16)

        if oni_dll.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr)) == 0:
            oni_frame = OniFrame.from_address(frame_ptr.value)
            data_size = oni_frame.width * oni_frame.height
            BufferType = ctypes.c_uint16 * data_size
            raw_buffer = BufferType.from_address(oni_frame.data)
            raw_depth = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(480, 640)

            # Read real distance at center pixel
            center_z_raw = raw_depth
            if center_z_raw.any() > 0:
                dynamic_shift_x = (FOCAL_LENGTH_PIXELS * LENS_BASELINE_MM) / center_z_raw
                dynamic_shift_x += 12
            else:
                dynamic_shift_x = 24

                # Warp layout matrix alignment configuration
            M = np.float32([[SCALE_F, 0, dynamic_shift_x], [0, SCALE_F, SHIFT_Y_STATIC]])
            warped_depth = cv2.warpAffine(raw_depth, M, (640, 480), flags=cv2.INTER_NEAREST)
            aligned_depth = cv2.flip(warped_depth, 1)

            # Lightweight morph filter to bridge fine micro-gaps smoothly
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            aligned_depth = cv2.morphologyEx(aligned_depth, cv2.MORPH_CLOSE, kernel)

            oni_dll.oniFrameRelease(frame_ptr)

        ret, raw_color_frame = color_cap.read()

        if ret and np.any(aligned_depth):
            color_frame = cv2.flip(raw_color_frame, 1)
            target_x, target_y = 320, 240

            # Read dynamic sliders to show parameters directly on the overlay display canvas
            current_exp = cv2.getTrackbarPos("IR Exposure", WINDOW_NAME)
            current_gain = cv2.getTrackbarPos("IR Gain", WINDOW_NAME)

            final_raw = aligned_depth[target_y, target_x]
            if final_raw > 0:
                distance_text = f"Distance: {final_raw / 10.0:.1f} cm"
            else:
                distance_text = "Distance: No Return Signal (0 mm)"

            # Draw visual tracking details
            cv2.circle(color_frame, (target_x, target_y), 6, (0, 0, 255), -1)
            cv2.putText(color_frame, distance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_frame, f"Exp: {current_exp} | Gain: {current_gain}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Generate depth map colors
            depth_scaled = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            blended_view = cv2.addWeighted(color_frame, 0.6, depth_colored, 0.4, 0)

            # Display live windows
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
