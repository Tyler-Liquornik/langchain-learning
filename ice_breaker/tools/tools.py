from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str) -> str:
    """Searches for a LinkedIn Profile Page URL from a person's name"""

    search = TavilySearchResults()
    return search.run(name)