import cv2
import numpy as np
import ctypes
import os
import time

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


STREAM_PROPERTY_MIRRORING = 3
STREAM_PROPERTY_AUTO_EXPOSURE = 23
STREAM_PROPERTY_EXPOSURE = 22


def set_stream_mirroring(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_MIRRORING, ctypes.byref(value), ctypes.sizeof(value))


def set_ir_auto_exposure(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.byref(value),
                                 ctypes.sizeof(value))


def set_ir_manual_exposure(stream_handle, exposure_value: int):
    set_ir_auto_exposure(stream_handle, False)
    value = ctypes.c_int(exposure_value)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_EXPOSURE, ctypes.byref(value), ctypes.sizeof(value))


# ===========================================================================
# 3. INITIALIZE HARDWARE CHANNELS
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

# ---------------------------------------------------------------------------
# HARDWARE UPDATE FOR DARK OBJECTS:
# Turn off auto-exposure and push manual exposure higher (e.g., 400).
# This gives the IR sensor more time to catch weak reflections from dark items.
# ---------------------------------------------------------------------------
set_ir_manual_exposure(stream_handle, 400)
print("[OPTIMIZED] Hardware exposure pushed high to scan dark surfaces.")

color_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
# 4. PARALLAX REALIGNMENT CONSTANTS
# ===========================================================================
FOCAL_LENGTH_PIXELS = 580.0
LENS_BASELINE_MM = 75.0
SHIFT_Y_STATIC = 4
SCALE_F = 1.02

CALIBRATION_ALPHA = 1.00
CALIBRATION_BETA = 0.0

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

            center_z_raw = raw_depth[240, 320]
            center_z_mm = (CALIBRATION_ALPHA * center_z_raw) + CALIBRATION_BETA

            if center_z_mm > 400:
                dynamic_shift_x = (FOCAL_LENGTH_PIXELS * LENS_BASELINE_MM) / center_z_mm
                dynamic_shift_x += 12
            else:
                dynamic_shift_x = 24

            M = np.float32([[SCALE_F, 0, dynamic_shift_x], [0, SCALE_F, SHIFT_Y_STATIC]])
            warped_depth = cv2.warpAffine(raw_depth, M, (640, 480), flags=cv2.INTER_NEAREST)
            # aligned_depth = cv2.flip(warped_depth, 1)
            aligned_depth = warped_depth

            # ---------------------------------------------------------------------------
            # SOFTWARE FILTER FOR DARK OBJECTS:
            # Apply a morphological "Closing" filter. This fills in small black holes (0 values)
            # caused by poor IR reflection by interpolating values from surrounding pixels.
            # ---------------------------------------------------------------------------
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            aligned_depth = cv2.morphologyEx(aligned_depth, cv2.MORPH_CLOSE, kernel)

            oni_dll.oniFrameRelease(frame_ptr)

        ret, raw_color_frame = color_cap.read()

        if ret and np.any(aligned_depth):
            color_frame = cv2.flip(raw_color_frame, 1)
            target_x, target_y = 320, 240

            # Measure distance
            final_raw = aligned_depth[target_y, target_x]
            if final_raw > 0:
                calibrated_dist_mm = (CALIBRATION_ALPHA * final_raw) + CALIBRATION_BETA
                distance_text = f"Distance: {calibrated_dist_mm / 10.0:.1f} cm"
            else:
                distance_text = f"Distance: Out of Range / Absorbed"

            # Render
            cv2.circle(color_frame, (target_x, target_y), 6, (0, 0, 255), -1)
            cv2.putText(color_frame, distance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            depth_scaled = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            blended_view = cv2.addWeighted(color_frame, 0.6, depth_colored, 0.4, 0)

            cv2.imshow("Astra Pro - Calibrated Video", color_frame)
            cv2.imshow("Astra Pro - RGB/Depth Alignment Blending", blended_view)

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
