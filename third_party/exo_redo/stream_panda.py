
#!/usr/bin/env python3
import socket
import json
import sys

if len(sys.argv) < 3:
    print("Usage: python3 joint_state_client_simple.py <host> <port>")
    print("Example: python3 joint_state_client_simple.py 0.tcp.ngrok.io 12345")
    sys.exit(1)

host = sys.argv[1]
port = int(sys.argv[2])

print(f"Connecting to {host}:{port}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))
print("Connected! Receiving joint states...\n")

buffer = ""
count = 0
while True:
    print("receiving data")
    data = sock.recv(4096).decode('utf-8')
    if not data:
        break
    buffer += data
    while '\n' in buffer:
        line, buffer = buffer.split('\n', 1)
        if line.strip():
            msg = json.loads(line)
            count += 1
            topic = msg.get('topic', 'unknown')
            pos = [f'{p:.3f}' for p in msg.get('positions', [])[:7]]
            print(f"[{count}] {topic}: {pos}")
