"""Test script for the ask-assistant API endpoint."""
import requests
import json

def test_ask_assistant():
    """Test the ask-assistant API endpoint."""
    url = "http://localhost:5000/ask-assistant"
    
    payload = {
        "query": "How do I troubleshoot this ticket?",
        "ticket_id": "12345",
        "focus_on_ticket_only": True,
        "current_incident_context": {
            "summary": "Unable to connect to VPN",
            "category": "Network",
            "priority": "High",
            "severity": "2",
            "latest_comments": "User reports connection drops intermittently"
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_ask_assistant()
