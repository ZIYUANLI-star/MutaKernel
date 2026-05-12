"""快速检查代理是否真的把流量发到海外。"""
import requests
import json

print("Step 1: 直连出口 IP")
try:
    r = requests.get(
        "https://ipinfo.io/json", timeout=10, proxies={"http": "", "https": ""}
    )
    d = r.json()
    print(f"  IP={d.get('ip')}  Country={d.get('country')}  Org={d.get('org')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nStep 2: 走系统代理出口 IP")
try:
    r = requests.get("https://ipinfo.io/json", timeout=10)
    d = r.json()
    print(f"  IP={d.get('ip')}  Country={d.get('country')}  Org={d.get('org')}")
    if d.get("country") == "CN":
        print("  >>> 代理未生效，仍是中国大陆出口 <<<")
    else:
        print("  >>> 代理已生效，出口在境外 <<<")
except Exception as e:
    print(f"  ERROR: {e}")
