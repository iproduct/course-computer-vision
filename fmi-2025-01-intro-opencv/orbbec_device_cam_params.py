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

# Define strict types for the native device get property signature
# int32_t oniDeviceGetProperty(void* device, int32_t propId, void* pData, int32_t* pDataSize)
oni_dll.oniDeviceGetProperty.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_int32)]
oni_dll.oniDeviceGetProperty.restype = ctypes.c_int32


# ---------------------------------------------------------------------------
# CRITICAL FIX: FORCE BYTE-PACK ALIGNMENT TO MATCH THE C++ COMPILER
# ---------------------------------------------------------------------------
class OBCameraParams(ctypes.Structure):
    _pack_ = 1  # Removes 64-bit operating system structural padding offsets
    _fields_ = [
        ("l_intr_p", ctypes.c_float * 4),  # Left IR camera intrinsics [fx, fy, cx, cy]
        ("r_intr_p", ctypes.c_float * 4),  # Right RGB webcam intrinsics [fx, fy, cx, cy]
        ("r2l_r", ctypes.c_float * 9),  # Rotation matrix [r00 to r22]
        ("r2l_t", ctypes.c_float * 3),  # Translation vector [t1, t2, t3]
        ("k", ctypes.c_float * 5),  # Distortion coefficients [k1, k2, k3, p1, p2]
        ("is_mirror", ctypes.c_int32)  # Hardware mirror flag
    ]


# The true Extended API identification index code for Orbbec cameras
OBEXTENSION_ID_CAM_PARAMS = 65539  # Hexadecimal equivalent: 0x10003

# ===========================================================================
# 2. RUN CORES LINK INITIALIZATION
# ===========================================================================

if oni_dll.oniInitialize(2) != 0:
    print("[ERROR] Failed to boot native OpenNI2 context.")
    exit()

device_handle = ctypes.c_void_p()
if oni_dll.oniDeviceOpen(None, ctypes.byref(device_handle)) != 0:
    print("[ERROR] No Astra camera detected by OpenNI2 drivers.")
    oni_dll.oniShutdown()
    exit()

print("[SUCCESS] Handshake secured. Querying Device via Extended API...")

# ===========================================================================
# 3. CONSTRUCT DATA MATRIX BOUNDS
# ===========================================================================
m_CamParams = OBCameraParams()
data_size = ctypes.c_int32(ctypes.sizeof(m_CamParams))

# Request property from device_handle using the packed structural allocation
status = oni_dll.oniDeviceGetProperty(
    device_handle,
    OBEXTENSION_ID_CAM_PARAMS,
    ctypes.pointer(m_CamParams),
    ctypes.pointer(data_size)
)

if status == 0:
    print("\n==================================================================")
    print("      SUCCESS: EXTRACTED NATIVE FACTORY CALIBRATION DATA")
    print("==================================================================\n")

    left_ir = list(m_CamParams.l_intr_p)
    right_rgb = list(m_CamParams.r_intr_p)
    translation_mm = list(m_CamParams.r2l_t)
    distortion_k = list(m_CamParams.k)

    print(f"[LEFT IR INTRA]    fx: {left_ir[0]:.2f}, fy: {left_ir[1]:.2f}, cx: {left_ir[2]:.2f}, cy: {left_ir[3]:.2f}")
    print(
        f"[RIGHT RGB INTRA]   fx: {right_rgb[0]:.2f}, fy: {right_rgb[1]:.2f}, cx: {right_rgb[2]:.2f}, cy: {right_rgb[3]:.2f}")
    print(
        f"[EXTRINSICS TRANS]  TX: {translation_mm[0]:.2f}mm, TY: {translation_mm[1]:.2f}mm, TZ: {translation_mm[2]:.2f}mm")
    print(
        f"[DISTORTION COEFF]  k1: {distortion_k[0]:.4f}, k2: {distortion_k[1]:.4f}, p1: {distortion_k[2]:.4f}, p2: {distortion_k[3]:.4f}")
    print(f"[HARDWARE MIRROR]   {m_CamParams.is_mirror}")
    print("\n------------------------------------------------------------------")

else:
    print(f"\n[FATAL ERROR] Extended API parameter request failed. Error Code: {status}")

# Safe Teardown
oni_dll.oniDeviceClose(device_handle)
oni_dll.oniShutdown()
print("\nSession safely closed.")