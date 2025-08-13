#!/usr/bin/env python3
"""
Test script for Ragie proxy endpoint
"""
import os
import requests
import urllib.parse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8080"  # Adjust if your server runs on different port
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")

def test_proxy_endpoint():
    """Test the Ragie proxy endpoint"""
    
    if not RAGIE_API_KEY:
        print("❌ RAGIE_API_KEY not found in environment")
        return False
    
    # Sample Ragie URL (this is the one you showed in the screenshot)
    test_ragie_url = "https://api.ragie.ai/documents/bc16c89f-73a3-48ce-89a9-5b382a15aa6a/chunks/9e33d0d5-6739-408e-9322-1d422a97504c/content?media_type=audio/mpeg"
    
    # Encode the URL for the proxy
    encoded_url = urllib.parse.quote(test_ragie_url)
    proxy_url = f"{BASE_URL}/api/proxy/ragie/stream?url={encoded_url}"
    
    print(f"🔗 Testing proxy URL: {proxy_url}")
    print(f"📋 Original Ragie URL: {test_ragie_url}")
    
    # You'll need to get a valid auth token from your browser's localStorage
    # For now, let's test without auth to see the error
    try:
        response = requests.get(proxy_url, timeout=10)
        print(f"📊 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code == 401:
            print("🔐 Authentication required (expected)")
            print("💡 You need to pass a valid Bearer token in the Authorization header")
            return True  # This is expected behavior
        elif response.status_code == 200:
            print("✅ Proxy working! Audio content received")
            print(f"📏 Content length: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Unexpected response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_direct_ragie_access():
    """Test direct access to Ragie (should fail without auth)"""
    test_ragie_url = "https://api.ragie.ai/documents/bc16c89f-73a3-48ce-89a9-5b382a15aa6a/chunks/9e33d0d5-6739-408e-9322-1d422a97504c/content?media_type=audio/mpeg"
    
    print(f"\n🔗 Testing direct Ragie access: {test_ragie_url}")
    
    try:
        response = requests.get(test_ragie_url, timeout=10)
        print(f"📊 Response status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 401:
            print("🔐 Direct access blocked (expected - no authentication)")
            return True
        else:
            print(f"❓ Unexpected response from direct access")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Ragie Proxy Configuration\n")
    
    print("=" * 50)
    print("TEST 1: Direct Ragie Access (should fail)")
    print("=" * 50)
    test_direct_ragie_access()
    
    print("\n" + "=" * 50)
    print("TEST 2: Proxy Endpoint (needs auth)")
    print("=" * 50)
    test_proxy_endpoint()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✅ If both tests show expected behavior, the proxy is configured correctly")
    print("🔐 You need to authenticate requests to the proxy endpoint")
    print("💡 The chat-summary.py should generate proxy URLs that your frontend can access with auth tokens")
