#!/usr/bin/env python3
"""
Diagnostic script to test SwissTopo Reframe API connectivity.

Run this to troubleshoot proxy/network issues.
"""

import os
import sys

def test_reframe_api():
    print("SwissTopo Reframe API Connectivity Test")
    print("=" * 60)
    
    # Check environment
    print("\n1. Environment Variables:")
    print(f"   HTTP_PROXY:  {os.environ.get('HTTP_PROXY', '(not set)')}")
    print(f"   HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '(not set)')}")
    print(f"   NO_PROXY:    {os.environ.get('NO_PROXY', '(not set)')}")
    print(f"   http_proxy:  {os.environ.get('http_proxy', '(not set)')}")
    print(f"   https_proxy: {os.environ.get('https_proxy', '(not set)')}")
    
    # Test URL
    test_url = "https://geodesy.geo.admin.ch/reframe/lv95towgs84"
    test_params = {
        'easting': '2600000',
        'northing': '1200000', 
        'altitude': '500',
        'format': 'json'
    }
    
    print(f"\n2. Test URL: {test_url}")
    print(f"   Parameters: {test_params}")
    
    # Try different methods
    import requests
    
    # Method 1: Default (uses environment proxy)
    print("\n3. Testing with default settings (environment proxy)...")
    try:
        response = requests.get(test_url, params=test_params, timeout=15)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ SUCCESS! Response: {data}")
            return True
        else:
            print(f"   ✗ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ Error: {type(e).__name__}: {e}")
    
    # Method 2: Bypass proxy
    print("\n4. Testing with proxy BYPASSED...")
    try:
        response = requests.get(
            test_url, 
            params=test_params, 
            timeout=15,
            proxies={'http': None, 'https': None}  # Bypass proxy
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ SUCCESS! Response: {data}")
            print("\n   → Solution: Set NO_PROXY=geo.admin.ch or use direct connection")
            return True
        else:
            print(f"   ✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {type(e).__name__}: {e}")
    
    # Method 3: Try with explicit proxy from environment
    print("\n5. Checking if proxy is required...")
    proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or \
                os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    
    if proxy_url:
        print(f"   Detected proxy: {proxy_url}")
        try:
            response = requests.get(
                test_url,
                params=test_params,
                timeout=15,
                proxies={'http': proxy_url, 'https': proxy_url}
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ SUCCESS with proxy! Response: {data}")
                return True
        except Exception as e:
            print(f"   ✗ Error with proxy: {e}")
    
    # Suggestions
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING SUGGESTIONS:")
    print("=" * 60)
    print("""
If you're behind a corporate proxy (e.g., EPFL):

Option 1: Bypass proxy for Swiss government domains
    export NO_PROXY=$NO_PROXY,geo.admin.ch,.admin.ch
    
Option 2: Configure proxy explicitly
    export HTTPS_PROXY=http://your-proxy:port
    
Option 3: Use pyproj-only mode (no network needed)
    Set in config.yaml: transformation_method: "pyproj"
    
Option 4: Check if VPN/firewall is blocking the connection

To test manually:
    curl "https://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=2600000&northing=1200000&altitude=500&format=json"
""")
    return False


def test_with_urllib():
    """Alternative test using urllib (different proxy handling)."""
    print("\n6. Testing with urllib (alternative)...")
    
    import urllib.request
    import json
    
    url = "https://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=2600000&northing=1200000&altitude=500&format=json"
    
    try:
        # This may use system proxy settings differently
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode())
            print(f"   ✓ SUCCESS with urllib! Response: {data}")
            return True
    except Exception as e:
        print(f"   ✗ Error: {type(e).__name__}: {e}")
        return False


if __name__ == '__main__':
    success = test_reframe_api()
    
    if not success:
        test_with_urllib()
    
    print("\n" + "=" * 60)
    if success:
        print("API connection successful! You can use transformation_method: 'reframe'")
    else:
        print("API connection failed. Use transformation_method: 'pyproj' as fallback")
    print("=" * 60)
