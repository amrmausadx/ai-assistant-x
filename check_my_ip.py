"""
Helper script to check your current IP address
Run this to find out what IP to add to config.py
"""

import requests
import socket

def get_external_ip():
    """Get your external/public IP address"""
    try:
        response = requests.get('https://api.ipify.org', timeout=5)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def get_local_ip():
    """Get your local network IP address"""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("IP ADDRESS CHECKER")
    print("=" * 60)
    print()
    
    print("🌐 Your EXTERNAL IP (public internet):")
    external_ip = get_external_ip()
    print(f"   {external_ip}")
    print()
    
    print("🏠 Your LOCAL IP (private network):")
    local_ip = get_local_ip()
    print(f"   {local_ip}")
    print()
    
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print()
    print("1. Open 'config.py' in your project")
    print("2. Add your IP to the ALLOWED_IPS list:")
    print()
    print("   ALLOWED_IPS = [")
    print("       '127.0.0.1',")
    if external_ip and not external_ip.startswith("Error"):
        print(f"       '{external_ip}',  # Your external IP")
    if local_ip and not local_ip.startswith("Error"):
        print(f"       '{local_ip}',   # Your local IP")
    print("   ]")
    print()
    print("3. Set ENABLE_IP_FILTERING = True")
    print("4. Restart your Flask application")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
