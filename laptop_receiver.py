import socket
import struct
import numpy as np

# Import the existing ML prediction function
# This keeps the ML pipeline entirely decoupled from the networking logic
from inference_demo import predict_machine_health

# ============================================================================
# CONFIGURATION
# ============================================================================
HOST = '0.0.0.0' # Listen on all available network interfaces
PORT = 8080      # Matches the port in esp32_firmware.ino

WINDOW_SIZE = 1024
PAYLOAD_SIZE = WINDOW_SIZE * 4 # 1024 floats * 4 bytes/float

def start_server():
    print(f"Starting ML Edge Node Receiver on {HOST}:{PORT}")
    print("Waiting for ESP32 to connect via Wi-Fi...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # Allow immediate port reuse
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        
        conn, addr = server.accept()
        with conn:
            print(f"ESP32 Connected from {addr}!")
            while True:
                # 1. Read the 4-byte Header
                header_bytes = conn.recv(4)
                if not header_bytes:
                    print("Connection lost.")
                    break
                    
                header = header_bytes.decode('utf-8', errors='ignore')
                
                # 2. Read the 4-byte Sample ID / Timestamp
                id_bytes = conn.recv(4)
                if len(id_bytes) < 4: break
                sample_id = struct.unpack('<I', id_bytes)[0]
                
                # 3. Read exactly 4096 bytes (1024 floats) of payload
                payload = b''
                while len(payload) < PAYLOAD_SIZE:
                    packet = conn.recv(PAYLOAD_SIZE - len(payload))
                    if not packet: break
                    payload += packet
                    
                if len(payload) < PAYLOAD_SIZE:
                    break
                    
                # Unpack the binary payload into a 1024-float tuple, then convert to Numpy
                raw_floats = struct.unpack(f'<{WINDOW_SIZE}f', payload)
                sensor_window = np.array(raw_floats, dtype=np.float32)
                
                # 4. Map Header to Sensor Type and Run Inference!
                print(f"--- Received Window #{sample_id} [{header}] ---")
                
                if header == "VIB:":
                    result = predict_machine_health(sensor_window, sensor_type='vibration')
                    print(f"Vibration ML Result: {result}")
                elif header == "SND:":
                    result = predict_machine_health(sensor_window, sensor_type='acoustic')
                    print(f"Acoustic ML Result:  {result}")
                else:
                    print(f"Unknown header received: {header}")

if __name__ == "__main__":
    start_server()
