import ctypes
import os

# ===========================================================================
# 1. SETUP HARDWARE PATHS & STRUCTURE DEFINITIONS
# ===========================================================================
OPENNI_REDIST_PATH = r"D:\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"

os.add_dll_directory(OPENNI_REDIST_PATH)
oni_dll = ctypes.cdll.LoadLibrary(os.path.join(OPENNI_REDIST_PATH, "OpenNI2.dll"))


# Define the native OpenNI2 C-struct for Device Information objects
class OniDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("uri", ctypes.c_char * 256),  # Dynamic system hardware URI path string
        ("vendor", ctypes.c_char * 256),  # Device vendor name (e.g., 'Orbbec')
        ("name", ctypes.c_char * 256),  # Device model description (e.g., 'Astra')
        ("usbVendorId", ctypes.c_uint16),  # USB Vendor identification block code
        ("usbProductId", ctypes.c_uint16)  # USB Product identification block code
    ]


# Define native types matching for the core function signatures
oni_dll.oniGetDeviceList.argtypes = [ctypes.POINTER(ctypes.POINTER(OniDeviceInfo)), ctypes.POINTER(ctypes.c_int)]
oni_dll.oniGetDeviceList.restype = ctypes.c_int

oni_dll.oniReleaseDeviceList.argtypes = [ctypes.POINTER(OniDeviceInfo)]
oni_dll.oniReleaseDeviceList.restype = ctypes.c_int

# ===========================================================================
# 2. RUN HARDWARE ENUMERATION HANDSHAKE
# ===========================================================================

# Initialize OpenNI2 system runtime engine context
if oni_dll.oniInitialize(2) != 0:
    print("[ERROR] Failed to boot native OpenNI2 context.")
    exit()

# Instantiate buffer pointers to receive the dynamic data lists from the driver
p_device_array = ctypes.POINTER(OniDeviceInfo)()
device_count = ctypes.c_int(0)

# Invoke the target list query function directly via OpenNI2.dll mapping hooks
print("Scanning active USB controllers for depth tracking modules...\n")
status = oni_dll.oniGetDeviceList(ctypes.byref(p_device_array), ctypes.byref(device_count))

if status == 0:
    total_found = device_count.value
    print(f"==================================================")
    print(f"  TOTAL DEVISES TRACKED: {total_found}")
    print(f"==================================================\n")

    # Loop over the retrieved C-array address pointers to extract information block elements
    for i in range(total_found):
        device_info = p_device_array[i]

        # Decode the raw byte string attributes back out to clean readable terminal elements
        device_uri = device_info.uri.decode('utf-8')
        vendor_name = device_info.vendor.decode('utf-8')
        model_name = device_info.name.decode('utf-8')

        print(f"[DEVICE INDEX {i}]")
        print(f"  -> Model Profile Name : {model_name}")
        print(f"  -> Hardware Vendor ID : {vendor_name}")
        print(f"  -> USB Vendor/Product : {hex(device_info.usbVendorId)} : {hex(device_info.usbProductId)}")
        print(f"  -> Unique System URI  : {device_uri}\n")

    # Free the temporary memory allocated by the driver array block
    oni_dll.oniReleaseDeviceList(p_device_array)
else:
    print(f"[ERROR] Device list query failed with operation exit code: {status}")

# Safe teardown
oni_dll.oniShutdown()
