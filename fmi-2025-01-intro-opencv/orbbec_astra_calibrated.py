import cv2
import numpy as np
import ctypes
import os
import time

# ===========================================================================
# 1. SETUP HARDWARE PATHS & DEFINE NATIVE STRUCTURES
# ===========================================================================

# Paste the absolute path to your extracted Orbbec OpenNI SDK 'Redist' folder here:
OPENNI_REDIST_PATH = r"D:\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"

# Add the directory to the Windows DLL lookup paths
os.add_dll_directory(OPENNI_REDIST_PATH)
oni_dll = ctypes.cdll.LoadLibrary(os.path.join(OPENNI_REDIST_PATH, "OpenNI2.dll"))


# Define OpenNI2 Native C-API Structure for Frame Buffers
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


# OpenNI 2 property constants for camera sensors
STREAM_PROPERTY_AUTO_EXPOSURE = 23
STREAM_PROPERTY_EXPOSURE = 22


# ===========================================================================
# 2. HARDWARE CONTROL UTILITY FUNCTIONS
# ===========================================================================

def set_ir_auto_exposure(stream_handle, enable: bool):
    """Enables or disables auto exposure on the infrared depth sensor."""
    value = ctypes.c_int(1 if enable else 0)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_AUTO_EXPOSURE,
                                 ctypes.byref(value), ctypes.sizeof(value))


def set_ir_manual_exposure(stream_handle, exposure_value: int):
    """Sets a fixed hardware exposure value for the IR sensor.
    Higher values capture data on dark/absorbing surfaces.
    Lower values prevent blowout on highly reflective white surfaces.
    """
    set_ir_auto_exposure(stream_handle, False)  # Must disable auto-exposure first
    value = ctypes.c_int(exposure_value)
    oni_dll.oniStreamSetProperty(stream_handle, STREAM_PROPERTY_EXPOSURE,
                                 ctypes.byref(value), ctypes.sizeof(value))


# ===========================================================================
# 3. INITIALIZE HARDWARE CHANNELS (OPENNI + UVC)
# ===========================================================================

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
oni_dll.oniDeviceCreateStream(device_handle, 1, ctypes.byref(stream_handle))  # 1 = SENSOR_DEPTH
oni_dll.oniStreamStart(stream_handle)
print("[SUCCESS] Hardware Depth Laser fired.")

# Configure initial sensor state (Start with Auto-Exposure to settle)
set_ir_auto_exposure(stream_handle, True)
current_exposure_setting = "AUTO"

# Connect to the split UVC webcam stream via DirectShow
color_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
# 4. CALIBRATION & SPATIAL ALIGNMENT CONSTANTS
# ===========================================================================
# Lens registration values to warp the IR image coordinates over the RGB lens coordinates
SHIFT_X = -22
SHIFT_Y = 4
SCALE_F = 1.02

# Baseline linear correction values for depth measurement drift
CALIBRATION_ALPHA = 1.00
CALIBRATION_BETA = 0.0

print("\n" + "=" * 50)
print("Astra Pro Calibration Stream Active.")
print("Controls:")
print("  Press 'a' -> Enable IR Auto Exposure")
print("  Press 'h' -> High IR Exposure (For Dark/Absorptive Surfaces)")
print("  Press 'l' -> Low IR Exposure (For Light/Shiny Surfaces)")
print("  Press 'q' -> Exit Program cleanly")
print("=" * 50 + "\n")

# ===========================================================================
# 5. DYNAMIC RGB-D PARALLAX REALIGNMENT LOOP
# ===========================================================================

# Baseline camera intrinsic estimates
FOCAL_LENGTH_PIXELS = 580.0  # Astra Pro IR lens focal approximation
LENS_BASELINE_MM = 75.0  # Physical lens gap spacing on chassis
SHIFT_Y_STATIC = 4  # Minor vertical lens offset

try:
    while True:
        frame_ptr = ctypes.c_void_p()
        aligned_depth = np.zeros((480, 640), dtype=np.uint16)

        # --- Read OpenNI Depth Frame Buffer ---
        if oni_dll.oniStreamReadFrame(stream_handle, ctypes.byref(frame_ptr)) == 0:
            oni_frame = OniFrame.from_address(frame_ptr.value)

            # Map raw 16-bit array from DLL memory space
            data_size = oni_frame.width * oni_frame.height
            BufferType = ctypes.c_uint16 * data_size
            raw_buffer = BufferType.from_address(oni_frame.data)
            raw_depth = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(480, 640)

            # 1. Sample the real-time distance value at the center pixel
            # We use an unaligned sample first to figure out the tracking distance Z
            center_z_raw = raw_depth[240, 320]

            # Apply your linear calibration correction layer
            center_z_mm = (CALIBRATION_ALPHA * center_z_raw) + CALIBRATION_BETA

            # 2. Calculate the Dynamic Parallax Shift based on physical distance
            if center_z_mm > 400:  # Ensure object is within real tracking bounds (> 40cm)
                # Parallax formula: (Focal Length * Physical Baseline) / Real Distance
                dynamic_shift_x = -(FOCAL_LENGTH_PIXELS * LENS_BASELINE_MM) / center_z_mm

                # Fine-tune anchor modifier if the alignment still drifts slightly
                dynamic_shift_x += -10
            else:
                # Fallback static shift if the sensor loses range track
                dynamic_shift_x = -22

                # 3. Apply the dynamic transformation matrix
            M = np.float32([[SCALE_F, 0, dynamic_shift_x], [0, SCALE_F, SHIFT_Y_STATIC]])
            aligned_depth = cv2.warpAffine(raw_depth, M, (640, 480), flags=cv2.INTER_NEAREST)

            # Release buffer handle inside C driver tier
            oni_dll.oniFrameRelease(frame_ptr)

        # --- Read Generic UVC Webcam RGB Frame ---
        ret, color_frame = color_cap.read()

        if ret and np.any(aligned_depth):
            target_x, target_y = 320, 240

            # Sample RGB values to track surface properties
            b, g, r = color_frame[target_y, target_x]
            perceived_brightness = 0.299 * r + 0.587 * g + 0.114 * b

            if perceived_brightness < 60:
                color_offset_modifier = -7.0
                color_type_text = "Dark Surface"
            elif perceived_brightness > 200:
                color_offset_modifier = 3.0
                color_type_text = "Bright Surface"
            else:
                color_offset_modifier = 0.0
                color_type_text = "Neutral Surface"

            # Re-read the calibrated value from the newly aligned center space
            final_raw = aligned_depth[target_y, target_x]
            if final_raw > 0:
                calibrated_dist_mm = (CALIBRATION_ALPHA * final_raw) + CALIBRATION_BETA + color_offset_modifier
                distance_text = f"Distance: {calibrated_dist_mm / 10.0:.1f} cm ({color_type_text})"
            else:
                distance_text = f"Distance: Out of Range / IR Shadow ({color_type_text})"

            # Render Overlays
            cv2.circle(color_frame, (target_x, target_y), 6, (0, 0, 255), -1)
            cv2.putText(color_frame, distance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_frame, f"Dynamic Shift X: {dynamic_shift_x:.1f} px", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Generate Colorized Depth Map Preview
            depth_scaled = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

            # Generate blended verification matrix view
            blended_view = cv2.addWeighted(color_frame, 0.6, depth_colored, 0.4, 0)

            # Display Windows
            cv2.imshow("Astra Pro - Calibrated Data Video", color_frame)
            cv2.imshow("Astra Pro - RGB/Depth Spatial Blending", blended_view)

        # Break loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('a'):
            set_ir_auto_exposure(stream_handle, True)
            current_exposure_setting = "AUTO"
            print(">> Switched to Hardware Auto-Exposure.")
        elif key == ord('h'):
            # Maximize exposure to capture dark, light-absorbing surfaces
            set_ir_manual_exposure(stream_handle, 350)
            current_exposure_setting = "MANUAL (HIGH - 350)"
            print(">> Fixed IR Exposure to HIGH (Optimized for dark targets).")
        elif key == ord('l'):
            # Minimize exposure to fix glare on white or shiny surfaces
            set_ir_manual_exposure(stream_handle, 65)
            current_exposure_setting = "MANUAL (LOW - 65)"
            print(">> Fixed IR Exposure to LOW (Optimized for reflective targets).")

finally:
    # --- Safe Native Resource Teardown ---
    print("\nClosing hardware communication pipelines...")
    oni_dll.oniStreamStop(stream_handle)
    oni_dll.oniStreamDestroy(stream_handle)
    oni_dll.oniDeviceClose(device_handle)
    oni_dll.oniShutdown()
    color_cap.release()
    cv2.destroyAllWindows()
    print("System safely shut down.")
