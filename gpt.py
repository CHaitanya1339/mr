import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define the API key and base URL
API_KEY = "eedf5fb5"
BASE_URL = "http://www.omdbapi.com/"
dataset = pd.read_csv('data.csv')


# Define a function to fetch the banner link for a movie
def get_banner_link(title):
    # Build the API request URL
    url = f"{BASE_URL}?apikey={API_KEY}&t={title}"
    # Send the API request and retrieve the JSON response
    response = requests.get(url)
    data = response.json()
    # Extract the poster URL from the response
    return data.get("Poster", "")

# Add a new column to the dataset with the banner links
dataset["banner_link"] = dataset["title"].apply(get_banner_link)

# Save the updated dataset to a new file
dataset.to_csv("movies_with_banners.csv", index=False)
