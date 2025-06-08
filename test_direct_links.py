"""Test script to verify direct ticket link functionality."""
from utils import generate_direct_ticket_link, verify_ticket_access_token
import os


def test_direct_links():
    """Test the direct ticket link generation and verification."""
    base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"
    test_ticket_ids = ["12345", "TICKET-001", "INC-789"]

    print("===== Testing Direct Ticket Links =====")

    for ticket_id in test_ticket_ids:
        # Generate a direct link
        direct_link = generate_direct_ticket_link(base_url, ticket_id)

        # Extract token from the link
        token = direct_link.split("/")[-1]

        # Verify token
        is_valid = verify_ticket_access_token(ticket_id, token)

        print(f"Ticket: {ticket_id}")
        print(f"  Direct Link: {direct_link}")
        print(f"  Token valid: {is_valid}")
        print()

    print("===== Testing Invalid Tokens =====")
    invalid_tokens = ["123", "abcdefg", "not-a-valid-token"]
    for token in invalid_tokens:
        is_valid = verify_ticket_access_token("12345", token)
        print(f"Token: {token}")
        print(f"  Valid: {is_valid}")
        print()


if __name__ == "__main__":
    test_direct_links()
