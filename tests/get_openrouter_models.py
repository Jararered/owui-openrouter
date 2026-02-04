import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

openrouter_endpoint = "https://openrouter.ai/api/v1"
headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
response = requests.get(f"{openrouter_endpoint}/models", headers=headers)

def get_openrouter_models():
    response = requests.get(f"{openrouter_endpoint}/models", headers=headers)
    return response.json()

if __name__ == "__main__":
    models = get_openrouter_models()
    print(models)

    # dump to file
    with open("tests/models.json", "w") as f:
        json.dump(models, f)