import os
import platform

def gpu_handshake():
    print("--- 2026 MLOps Station Handshake ---")
    
    # 1. Check OS and Python
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    # 2. Check for the AMD HIP SDK (The Muscle)
    # This path is the default for the installer you just ran
    hip_path = r"C:\Program Files\AMD\ROCm\6.4\bin"
    if os.path.exists(hip_path):
        print(f"✅ HANDSHAKE SUCCESS: AMD HIP SDK found at {hip_path}")
    else:
        print("❌ HANDSHAKE FAILED: Cannot find HIP SDK. Did it install to C:?")

    # 3. Check for the Warehouse (D: Drive)
    mlops_path = r"D:\MLOps\input_data"
    if os.path.exists(mlops_path):
        print(f"✅ STORAGE SUCCESS: D: Drive Warehouse is online.")
    else:
        print("❌ STORAGE FAILED: Check your D: drive folders.")

if __name__ == "__main__":
    gpu_handshake()