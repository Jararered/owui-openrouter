import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_openrouter_models():
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://openwebui.com",
            "X-Title": "OpenWebUI",
            "Content-Type": "application/json",
        },
    )
    return response.json()["data"]


if __name__ == "__main__":
    models = get_openrouter_models()
    print(models)

    print(f"Found {len(models)} models")

    # dump to file
    with open("tests/models.json", "w") as f:
        json.dump(models, f)
