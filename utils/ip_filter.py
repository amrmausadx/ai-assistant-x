"""
IP Filtering Middleware for Flask Application
Protects against unauthorized access and automated scanning
"""

from flask import request, jsonify
from functools import wraps
import ipaddress
import config

def is_local_network_ip(ip_str):
    """Check if an IP address is from a local/private network"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False

def get_real_ip():
    """
    Get the real IP address from the request.
    Handles proxies and load balancers by checking common headers.
    """
    # Check common proxy headers in order of preference
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    elif request.headers.get('X-Forwarded-For'):
        # X-Forwarded-For can contain multiple IPs, get the first one (client IP)
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('CF-Connecting-IP'):  # Cloudflare
        return request.headers.get('CF-Connecting-IP')
    else:
        return request.remote_addr

def is_ip_allowed(ip_address):
    """
    Check if an IP address is allowed to access the application
    """
    # If IP filtering is disabled, allow all
    if not config.ENABLE_IP_FILTERING:
        return True
    
    # Check if IP is in whitelist
    if ip_address in config.ALLOWED_IPS:
        return True
    
    # Check if local network IPs are allowed
    if config.ALLOW_LOCAL_NETWORK and is_local_network_ip(ip_address):
        return True
    
    return False

def require_ip_whitelist(f):
    """
    Decorator to protect routes with IP filtering
    Usage: @require_ip_whitelist
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = get_real_ip()
        
        if not is_ip_allowed(client_ip):
            print(f"🚫 Access denied for IP: {client_ip}")
            return jsonify({
                'error': 'Access Denied',
                'message': 'Your IP address is not authorized to access this resource.'
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

def init_ip_filtering(app):
    """
    Initialize IP filtering for the Flask app
    This adds a before_request handler to check all requests
    """
    @app.before_request
    def check_ip():
        # Skip IP check for static files
        if request.path.startswith('/static/'):
            return None
        
        client_ip = get_real_ip()
        
        if not is_ip_allowed(client_ip):
            print(f"🚫 Blocked request from IP: {client_ip} to {request.path}")
            return jsonify({
                'error': 'Access Denied',
                'message': 'Your IP address is not authorized to access this application.'
            }), 403
        
        # Log allowed access (optional, comment out if too verbose)
        if config.ENABLE_IP_FILTERING:
            print(f"✅ Allowed request from IP: {client_ip} to {request.path}")
        
        return None

def log_suspicious_activity(ip_address, path, user_agent=None):
    """
    Log suspicious activity for security monitoring
    You can extend this to write to a file or database
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"[{timestamp}] SUSPICIOUS: IP={ip_address}, Path={path}"
    if user_agent:
        log_entry += f", User-Agent={user_agent}"
    
    print(log_entry)
    
    # Optionally write to a log file
    try:
        with open('security.log', 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"Failed to write to security log: {e}")
