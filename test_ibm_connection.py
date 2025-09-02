#!/usr/bin/env python3
"""Test IBM Quantum connectivity from your environment."""

import os
import socket
import requests
from urllib.parse import urlparse

# Test DNS resolution and connectivity
def test_connectivity():
    hosts = [
        "auth.quantum-computing.ibm.com",
        "quantum-computing.ibm.com",
        "cloud.ibm.com",
        "google.com"  # Control test
    ]
    
    print("Testing DNS resolution and connectivity...")
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"✓ {host} -> {ip}")
            
            # Try HTTP request
            try:
                resp = requests.get(f"https://{host}", timeout=5)
                print(f"  HTTP status: {resp.status_code}")
            except Exception as e:
                print(f"  HTTP failed: {type(e).__name__}")
        except socket.gaierror:
            print(f"✗ {host} -> DNS resolution failed")
        except Exception as e:
            print(f"✗ {host} -> Error: {e}")
    
    print("\nTesting with direct Qiskit...")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.environ.get("IBM_QUANTUM_API_TOKEN", "y2WlLfLb-btrG9QFdA9C0VVnOHhgqRMECqULsrD_1n57")
        service = QiskitRuntimeService(token=token, channel="ibm_quantum")
        print("✓ QiskitRuntimeService initialized")
        backends = service.backends()
        print(f"  Available backends: {[b.name for b in backends[:5]]}...")
    except Exception as e:
        print(f"✗ QiskitRuntimeService failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_connectivity()
