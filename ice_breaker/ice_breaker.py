import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Call this once at the start
    load_dotenv()

    # API Key value accessible assuming it's in .env
    print(os.environ['SAMPLE_API_KEY'])