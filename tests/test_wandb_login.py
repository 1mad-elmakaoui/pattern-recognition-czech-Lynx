"""Test wandb login configuration - gut check for account setup."""
import os
import pytest
import wandb
import dotenv

dotenv.load_dotenv()

def test_wandb_login():
    """
    Test that wandb can be logged in.
    """
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        pytest.fail(
            "WANDB_API_KEY environment variable is not set.\n"
            "To fix this:\n"
            "  1. Get your API key from: https://wandb.ai/authorize\n"
            "  2. Set it as an environment variable:\n"
            "     export WANDB_API_KEY='your-api-key-here'\n"
            "     or \n"
            "     add it to your .env file:\n"
            "     WANDB_API_KEY='your-api-key-here'"
        )
    
    try:
        result = wandb.login(key=api_key)
        if not result:
            pytest.fail(
                "wandb.login() returned False. Your API key may be invalid.\n"
                "To fix this, verify your API key at: https://wandb.ai/authorize\n"
            )
    except Exception as e:
        raise Exception(f"Unexpected error during wandb login: {e}")
    
    print("✓ wandb login successful")
