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


# Core OpenNI2 Fixed Global Hardware Mapping Constants
# Reference: http://ros.org
ONI_SENSOR_DEPTH = 1
ONI_SENSOR_IR = 2

STREAM_PROPERTY_MIRRORING = 7  # FIXED ENUM: Checked against core headers
STREAM_PROPERTY_AUTO_EXPOSURE = 101  # FIXED ENUM: Checked against core headers
STREAM_PROPERTY_EXPOSURE = 102  # FIXED ENUM: Checked against core headers
STREAM_PROPERTY_GAIN = 103  # FIXED ENUM: Checked against core headers


def set_stream_mirroring(stream_handle, enable: bool):
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_MIRRORING, ctypes.byref(value), ctypes.sizeof(value))


def set_ir_hardware_controls(ir_stream_handle, auto_exp: bool, exp_val: int, gain_val: int):
    """Pushes the slider parameters to the physical active laser lens layer."""
    if ir_stream_handle is None:
        return

    # 1. Toggle hardware Auto Exposure
    ae_val = ctypes.c_int(1 if auto_exp else 0)
    oni_dll.oniStreamSetProperty(ir_stream_handle, STREAM_PROPERTY_AUTO_EXPOSURE, ctypes.byref(ae_val),
                                 ctypes.sizeof(ae_val))

    # 2. Inject manual parameters if auto exposure is disabled
    if not auto_exp:
        e_val = ctypes.c_int(exp_val)
        oni_dll.oniStreamSetProperty(ir_stream_handle, STREAM_PROPERTY_EXPOSURE, ctypes.byref(e_val),
                                     ctypes.sizeof(e_val))

        g_val = ctypes.c_int(gain_val)
        oni_dll.oniStreamSetProperty(ir_stream_handle, STREAM_PROPERTY_GAIN, ctypes.byref(g_val), ctypes.sizeof(g_val))


# ===========================================================================
# 2. INITIALIZE HARDWARE ENDPOINTS (DEPTH + IR STREAM CORES)
# ===========================================================================

if oni_dll.oniInitialize(2) != 0:
    print("[ERROR] Failed to initialize native OpenNI2 DLL context.")
    exit()

device_handle = ctypes.c_void_p()
if oni_dll.oniDeviceOpen(None, ctypes.byref(device_handle)) != 0:
    print("[ERROR] No Orbbec camera detected by OpenNI2 drivers.")
    oni_dll.oniShutdown()
    exit()

# Stream A: Depth Processing Stream
depth_stream_handle = ctypes.c_void_p()
oni_dll.oniDeviceCreateStream(device_handle, ONI_SENSOR_DEPTH, ctypes.byref(depth_stream_handle))
oni_dll.oniStreamStart(depth_stream_handle)
set_stream_mirroring(depth_stream_handle, False)

# Stream B: Infrared Control Stream (Must remain open to receive manual exposure values)
ir_stream_handle = ctypes.c_void_p()
has_ir_control = False
if oni_dll.oniDeviceCreateStream(device_handle, ONI_SENSOR_IR, ctypes.byref(ir_stream_handle)) == 0:
    oni_dll.oniStreamStart(ir_stream_handle)
    set_stream_mirroring(ir_stream_handle, False)
    has_ir_control = True
    print("[SUCCESS] Hardware mapping linked directly to native IR camera controller.")

# Connect to the split UVC webcam stream via DirectShow
color_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
# 3. SETUP MASTER TRACKBAR PANEL WITH CORRECT HARDWARE SCALES
# ===========================================================================
WINDOW_NAME = "Astra Pro - Master Control Panel"
cv2.namedWindow(WINDOW_NAME)


def nothing(val):
    pass


# Sliders mapped cleanly to calibrated firmware limits
cv2.createTrackbar("Auto Exposure (0=Off, 1=On)", WINDOW_NAME, 0, 1, nothing)
cv2.createTrackbar("IR Exposure Scale", WINDOW_NAME, 400, 2000, nothing)  # SCALE FIXED: Range expanded to 2000
cv2.createTrackbar("IR Gain Scale", WINDOW_NAME, 8, 31, nothing)  # SCALE FIXED: Range locked to native 0-31 bounds

# Spatial alignment sliders
cv2.createTrackbar("Shift X Offset", WINDOW_NAME, 62, 100, nothing)
cv2.createTrackbar("Shift Y Offset", WINDOW_NAME, 54, 100, nothing)
cv2.createTrackbar("Scale FX (x100)", WINDOW_NAME, 102, 150, nothing)

# ===========================================================================
# 4. EXECUTION RUNTIME STREAMING LOOP
# ===========================================================================
SCALE_F = 1.02

try:
    while True:
        frame_ptr = ctypes.c_void_p()
        aligned_depth = np.zeros((480, 640), dtype=np.uint16)

        # Pull trackbar positions down to processing parameters live
        ae_flag = cv2.getTrackbarPos("Auto Exposure (0=Off, 1=On)", WINDOW_NAME) == 1
        ir_exposure = cv2.getTrackbarPos("IR Exposure Scale", WINDOW_NAME)
        ir_gain = cv2.getTrackbarPos("IR Gain Scale", WINDOW_NAME)

        cur_shift_x = cv2.getTrackbarPos("Shift X Offset", WINDOW_NAME) - 50
        cur_shift_y = cv2.getTrackbarPos("Shift Y Offset", WINDOW_NAME) - 50
        cur_scale_fx = cv2.getTrackbarPos("Scale FX (x100)", WINDOW_NAME) / 100.0

        # Push controls to the IR stream endpoint
        if has_ir_control:
            set_ir_hardware_controls(ir_stream_handle, ae_flag, ir_exposure, ir_gain)

        # Read frame buffer data
        if oni_dll.oniStreamReadFrame(depth_stream_handle, ctypes.byref(frame_ptr)) == 0:
            oni_frame = OniFrame.from_address(frame_ptr.value)
            data_size = oni_frame.width * oni_frame.height
            BufferType = ctypes.c_uint16 * data_size
            raw_buffer = BufferType.from_address(oni_frame.data)
            raw_depth = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(480, 640)

            # Spatial Calibration Matrix
            M = np.float32([[cur_scale_fx, 0, cur_shift_x], [0, cur_scale_fx, cur_shift_y]])
            warped_depth = cv2.warpAffine(raw_depth, M, (640, 480), flags=cv2.INTER_NEAREST)

            # Unified horizontal flip correction applied inside the software matrix space
            aligned_depth = cv2.flip(warped_depth, 1)

            # Morphology closing filter to smooth edge detection noise
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
            cv2.putText(color_frame, f"Exp: {ir_exposure} | Gain: {ir_gain}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Generate depth map colors
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
    oni_dll.oniStreamStop(depth_stream_handle)
    oni_dll.oniStreamDestroy(depth_stream_handle)
    if has_ir_control:
        oni_dll.oniStreamStop(ir_stream_handle)
        oni_dll.oniStreamDestroy(ir_stream_handle)
    oni_dll.oniDeviceClose(device_handle)
    oni_dll.oniShutdown()
    color_cap.release()
    cv2.destroyAllWindows()