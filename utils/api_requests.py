import os
import logging
import requests
import time
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

def call_groq_api(prompt):
    """Calls Groq's API with throttling and retrieves a response."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("GROQ_API_KEY is not set. Please provide a valid API key.")
            return None

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5000
        }

        while True:
            response = requests.post(GROQ_API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                json_response = response.json()
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    return json_response["choices"][0]["message"]["content"].strip()
                else:
                    logging.error("No choices returned from Groq API.")
                    return None

            elif response.status_code == 429:  # Rate limit reached
                error_info = response.json().get("error", {})
                wait_time = error_info.get("message", "").split("Please try again in ")[-1].split("s")[0]
                try:
                    wait_time = float(wait_time)
                except ValueError:
                    wait_time = 60  # Default wait time
                logging.warning(f"Rate limit reached. Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
                continue

            elif response.status_code == 401:
                logging.error("Invalid API Key. Please check your GROQ_API_KEY.")
                return None

            elif response.status_code == 404:
                logging.error("Model not found. Please verify the model name is correct.")
                return None

            else:
                logging.error(f"Groq API error {response.status_code}: {response.text}")
                return None

    except Exception as e:
        logging.error(f"Groq API call failed: {e}")
        return None
