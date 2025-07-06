import os
import requests
import pprint
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """Manually scrape information from LinkedIn profiles,
    mock setup with GitHub Gist for testing only"""

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/Tyler-Liquornik/0fd5161c15205545470a357ccb56162d/raw/215e217fda558fc83931f48df56c324e4353a428/satya-nadella-scrapin.json"
        response = requests.get(linkedin_profile_url,timeout=10)
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {"apikey": os.environ["SCRAPIN_API_KEY"], "linkedInUrl": linkedin_profile_url}
        response = requests.get(api_endpoint, params=params, timeout=10)

    person_data = response.json().get("person")
    final_data = {
        k: v
        for k, v in person_data.items()
        if v not in ([], "", None)

        # We plan to send this data to an LLM, and every token costs $$$
        # In this example, we don't care about certifications and want to lighten our payload
        and k not in ["certifications"]
    }

    return final_data

# For testing, we can add an entry point to this script
if __name__ == "__main__":
    pprint.pprint(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/satyanadella/"
        )
    )