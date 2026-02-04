#!/usr/bin/env python3
"""
Script to verify OpenRouter API key and retrieve all models.

This script:
1. Loads OPENROUTER_API_KEY from .env file
2. Verifies the API key by fetching models from OpenRouter using OpenrouterAPI
3. Outputs all models to openrouter_models.json
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from owui_openrouter import OpenrouterAPI, OpenrouterAPIConfig


def load_api_key() -> str:
    """Load API key from .env file."""
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        sys.exit(1)
    
    return api_key


def get_raw_models(api: OpenrouterAPI) -> dict:
    """
    Get raw models data from OpenRouter using the OpenrouterAPI instance.
    
    Returns:
        dict: Raw response containing models data
    """
    if not api.config.api_key:
        print("ERROR: API key not configured")
        sys.exit(1)
    
    try:
        print("Fetching models from OpenRouter...")
        headers = api.get_auth_headers()
        response = requests.get(f"{api.api_base}/models", headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Successfully retrieved {len(data.get('data', []))} models")
        return data
        
    except requests.exceptions.HTTPError as e:
        error_data = api.handle_api_error(e.response)
        if e.response.status_code == 401:
            print("ERROR: Invalid API key (401 Unauthorized)")
        elif e.response.status_code == 403:
            print("ERROR: API key does not have permission (403 Forbidden)")
        else:
            print(f"ERROR: HTTP {e.response.status_code}: {e.response.reason}")
        
        if error_data.get("error"):
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        sys.exit(1)
        
    except requests.exceptions.RequestException as e:
        error_data = api.handle_request_exception(e)
        print(f"ERROR: Request failed: {error_data.get('error', {}).get('message', str(e))}")
        sys.exit(1)
        
    except Exception as e:
        error_data = api.handle_request_exception(e)
        print(f"ERROR: Unexpected error: {error_data.get('error', {}).get('message', str(e))}")
        sys.exit(1)


def save_models_to_json(models_data: dict, output_file: str = "openrouter_models.json"):
    """Save models data to JSON file."""
    # Get the directory where this test file is located
    test_dir = Path(__file__).parent
    output_path = test_dir / output_file
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(models_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Models saved to {output_path}")
        print(f"  Total models: {len(models_data.get('data', []))}")
    except Exception as e:
        print(f"ERROR: Failed to save JSON file: {str(e)}")
        sys.exit(1)


def main():
    """Main function."""
    print("OpenRouter API Key Verification Test")
    print("=" * 50)
    
    # Load API key
    api_key = load_api_key()
    print(f"API key loaded from .env file")
    print(f"Key prefix: {api_key[:10]}...")
    
    # Create OpenrouterAPI instance
    config = OpenrouterAPIConfig(api_key=api_key)
    api = OpenrouterAPI(config)
    
    # Get raw models data
    models_data = get_raw_models(api)
    
    # Save to JSON file
    save_models_to_json(models_data)
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
