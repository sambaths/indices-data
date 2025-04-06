import requests
import pandas as pd
from collections import defaultdict

# Define headers to mimic a browser
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/135.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-AU,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Function to fetch data from a given NSE API endpoint
def fetch_nse_data(url, session):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return []

# Main function to aggregate data from multiple endpoints
def main():
    session = requests.Session()
    session.headers.update(HEADERS)
    # Access the NSE homepage to establish the session and obtain cookies
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception as e:
        print(f"Error accessing NSE homepage: {e}")
        return

    # Define the NSE API endpoints and their corresponding categories
    endpoints = {
        "Most Active by Value": "https://www.nseindia.com/api/live-analysis-most-active-securities?index=value",
        "Volume Gainers": "https://www.nseindia.com/api/live-analysis-volume-gainers",
        "Large Deals": "https://www.nseindia.com/api/snapshot-capital-market-largedeal",
        "Most Active by Volume": "https://www.nseindia.com/api/live-analysis-most-active-securities?index=volume",
        "Top Gainers": "https://www.nseindia.com/api/live-analysis-variations?index=gainers",
        "Top Losers": "https://www.nseindia.com/api/live-analysis-variations?index=loosers",
    }

    # Fetch data from each endpoint
    data = {category: fetch_nse_data(url, session) for category, url in endpoints.items()}

    data['Most Active by Value'] = data['Most Active by Value']['data']
    data['Most Active by Volume'] = data['Most Active by Volume']['data']
    data['Volume Gainers'] = data['Volume Gainers']['data']

    # Large Deals
    data['Bulk Deals Data'], data['Block Deals Data'], data['Short Deals Data'] = data['Large Deals']['BULK_DEALS_DATA'], data['Large Deals']['BLOCK_DEALS_DATA'], data['Large Deals']['SHORT_DEALS_DATA']
    del data['Large Deals']

    # Gainers
    data['Top Gainers'] = data['Top Gainers']['allSec']['data']

    # Loosers
    data['Top Losers'] = data['Top Losers']['allSec']['data']

    # Aggregate reasons for each stock's inclusion
    stock_reasons = defaultdict(set)
    all_data = pd.DataFrame()
    for category, entries in data.items():
      category_df = pd.DataFrame()
      for entry in entries:
          temp = pd.DataFrame(entry, index=[0])
          temp['Reason'] = category
          temp['relavant_columns'] = ", ".join(entry.keys())
          category_df = pd.concat([category_df, temp])
      all_data = pd.concat([all_data, category_df])

    import os
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save to CSV
    output_file = "data/active_stocks_with_reasons.csv"
    all_data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
