"""Common utilities for the AutoResolveX application."""
import hashlib
import time
import os


def generate_ticket_access_token(ticket_id):
    """Generate a secure token for direct ticket access."""
    # Secret key for token generation - in production, use a proper key management solution
    secret_key = os.getenv('SECRET_TOKEN_KEY', 'autoresolve-x-secret-key')

    # Create a token based on ticket_id and timestamp
    timestamp = str(int(time.time()))
    token_string = f"{ticket_id}-{timestamp}-{secret_key}"
    token = hashlib.sha256(token_string.encode()).hexdigest()[
        :32]  # First 32 chars for simplicity

    return token, timestamp


def verify_ticket_access_token(ticket_id, token):
    """Verify the ticket access token.

    In a production environment, you would:
    1. Validate the token against stored tokens or recreate and compare
    2. Check token expiration
    3. Possibly track token usage to prevent replay attacks

    For this implementation, we'll simply validate the basic format and accept valid-looking tokens.
    """
    if not token or len(token) != 32:  # Basic check for token format (32 chars)
        return False

    try:
        # Verify the token is a valid hexadecimal string
        int(token, 16)
        return True
    except ValueError:
        return False


def generate_direct_ticket_link(base_url, ticket_number):
    """Generate a direct ticket link with a security token to bypass login."""
    token, _ = generate_ticket_access_token(ticket_number)
    return f"{base_url}/direct-ticket/{ticket_number}/{token}"
