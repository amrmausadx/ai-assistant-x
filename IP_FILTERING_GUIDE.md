# IP Filtering Configuration Guide

## Overview
This application now includes IP filtering to protect against unauthorized access and automated scanning attempts.

## Quick Start

### 1. **Add Your IP Address**

Open `config.py` and add your IP address to the `ALLOWED_IPS` list:

```python
ALLOWED_IPS = [
    '127.0.0.1',        # Localhost
    '::1',              # IPv6 localhost
    '203.0.113.45',     # Your real IP address (example)
]
```

### 2. **Find Your Current IP Address**

**Windows:**
```powershell
# External IP (what the internet sees)
(Invoke-WebRequest -Uri "https://api.ipify.org").Content

# Local network IP
ipconfig | Select-String "IPv4"
```

**Linux/Mac:**
```bash
# External IP
curl https://api.ipify.org

# Local network IP
hostname -I
```

### 3. **Enable IP Filtering**

In `config.py`, set:
```python
ENABLE_IP_FILTERING = True
```

### 4. **Restart the Application**

Stop and restart your Flask application for changes to take effect.

## Configuration Options

### `ALLOWED_IPS`
List of IP addresses that are allowed to access the application.

```python
ALLOWED_IPS = [
    '127.0.0.1',        # Localhost
    '192.168.1.100',    # Home network
    '10.0.0.50',        # Office network
    '203.0.113.45',     # Remote IP
]
```

### `ENABLE_IP_FILTERING`
- `True`: Only IPs in `ALLOWED_IPS` (and local network if enabled) can access
- `False`: All IPs can access (default for development)

### `ALLOW_LOCAL_NETWORK`
- `True`: Allows all local/private network IPs (192.168.x.x, 10.x.x.x, 172.16.x.x)
- `False`: Only IPs explicitly listed in `ALLOWED_IPS` are allowed

## Usage Scenarios

### Development (Default)
```python
ENABLE_IP_FILTERING = False  # Allow all IPs
ALLOW_LOCAL_NETWORK = True
```

### Production (Recommended)
```python
ENABLE_IP_FILTERING = True   # Restrict access
ALLOW_LOCAL_NETWORK = False  # Only specific IPs
ALLOWED_IPS = ['your.public.ip.here']
```

### Home/Office Network
```python
ENABLE_IP_FILTERING = True
ALLOW_LOCAL_NETWORK = True   # Allow all devices on local network
ALLOWED_IPS = ['your.public.ip.here']  # Plus your remote IP
```

## Behind a Proxy or Load Balancer

If your application is behind a proxy (like Nginx, Cloudflare, or a load balancer), the IP filtering automatically checks these headers:

- `X-Real-IP`
- `X-Forwarded-For`
- `CF-Connecting-IP` (Cloudflare)

No additional configuration needed!

## Security Logging

Blocked requests are automatically logged to:
- **Console**: Real-time blocking notifications
- **File**: `security.log` (created automatically)

Example log entry:
```
[2025-12-11 19:00:00] SUSPICIOUS: IP=192.0.2.1, Path=/admin
```

## Testing

### Test if IP filtering is working:

1. **Enable filtering:**
   ```python
   ENABLE_IP_FILTERING = True
   ALLOWED_IPS = ['127.0.0.1']  # Only localhost
   ```

2. **Access from localhost** → Should work ✅
3. **Access from another device** → Should be blocked 🚫

### Expected blocked response:
```json
{
  "error": "Access Denied",
  "message": "Your IP address is not authorized to access this application."
}
```

## Troubleshooting

### "I can't access the application after enabling filtering"

1. Check your current IP:
   ```powershell
   (Invoke-WebRequest -Uri "https://api.ipify.org").Content
   ```

2. Add it to `ALLOWED_IPS` in `config.py`

3. Restart the application

### "IP filtering isn't working"

1. Verify `ENABLE_IP_FILTERING = True` in `config.py`
2. Check the console output when starting the app
3. Look for "🔒 IP Filtering: ENABLED" message

### "Getting blocked on local network"

Set `ALLOW_LOCAL_NETWORK = True` in `config.py`

## Advanced: Protecting Specific Routes Only

If you want to protect only certain routes (not the entire app), you can use the `@require_ip_whitelist` decorator:

```python
from utils.ip_filter import require_ip_whitelist

@app.route('/admin')
@require_ip_whitelist
def admin():
    return "Admin panel"
```

## Best Practices

1. **Always keep `127.0.0.1` in `ALLOWED_IPS`** for local testing
2. **Use `ENABLE_IP_FILTERING = False` during development** to avoid lockouts
3. **Enable filtering in production** to protect against automated attacks
4. **Regularly review `security.log`** for suspicious activity
5. **Update `ALLOWED_IPS` when your IP changes** (if you have a dynamic IP)

## Example Configurations

### Personal Development Server
```python
ENABLE_IP_FILTERING = True
ALLOW_LOCAL_NETWORK = True
ALLOWED_IPS = [
    '127.0.0.1',
    '203.0.113.45',  # Your home IP
]
```

### Team Development Server
```python
ENABLE_IP_FILTERING = True
ALLOW_LOCAL_NETWORK = True
ALLOWED_IPS = [
    '127.0.0.1',
    '203.0.113.45',  # Team member 1
    '203.0.113.46',  # Team member 2
    '203.0.113.47',  # Team member 3
]
```

### Public Demo (No Filtering)
```python
ENABLE_IP_FILTERING = False
ALLOW_LOCAL_NETWORK = True
```

---

**Need Help?** Check the console output when starting the application for IP filtering status and allowed IPs.
