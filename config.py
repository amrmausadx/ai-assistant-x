"""
Configuration file for the AI Assistant application
"""

# ============================================================================
# ACCESS CONTROL CONFIGURATION
# ============================================================================

# Set to True to enable IP filtering (restrict access), False to allow anyone
ENABLE_IP_FILTERING = False  # Currently: Anyone can access the site

# Client IP Whitelist (only used when ENABLE_IP_FILTERING = True)
# These are the IPs that are ALLOWED to connect TO your server
ALLOWED_IPS = [
    '127.0.0.1',        # Localhost
    '::1',              # IPv6 localhost
    # Add client IPs here if you want to restrict access:
    # '203.0.113.45',   # Example: Specific client IP
]

# Set to True to allow all local network IPs (192.168.x.x, 10.x.x.x, etc.)
ALLOW_LOCAL_NETWORK = True

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Server IP addresses (where Flask is running)
# These are YOUR server's network interfaces
SERVER_IPS = [
    '193.227.24.34',      # Server IP #1
    '196.218.189.179',    # Server IP #2
]

# Flask Configuration
DEBUG = False

# HOST: Which network interface to bind to
# - '0.0.0.0' = Listen on ALL network interfaces (recommended for public access)
# - '127.0.0.1' = Only localhost (local access only)
# - Specific IP = Only that network interface
HOST = '0.0.0.0'  # Listen on all interfaces (allows access from any IP)

PORT = 5000
THREADED = True
