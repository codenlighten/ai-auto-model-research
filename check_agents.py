"""
Check what agents are already registered
"""
import requests

BASE_URL = "https://lumenbridge.xyz"
USER_ID = "ai-research-lab"

url = f"{BASE_URL}/api/agents/my-agents/{USER_ID}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"\n✅ Found {data.get('count', 0)} agents:\n")
    for agent in data.get('agents', []):
        print(f"  • {agent['name']}")
        print(f"    Description: {agent.get('description', 'N/A')[:80]}...")
        print(f"    Created: {agent.get('createdAt', 'N/A')}\n")
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)
