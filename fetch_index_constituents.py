import requests, os
import urllib.parse
import pandas as pd
from tqdm import tqdm

# Define headers to mimic a browser.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

def get_nse_master():
    """
    Retrieve the NSE master equity data from the official API.
    Returns the JSON data (a list of instruments) and a session object.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    # First hit the NSE homepage to obtain cookies.
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=5)
    except Exception as e:
        print(f"Error accessing NSE homepage: {e}")
    master_url = "https://www.nseindia.com/api/equity-master"
    try:
        response = session.get(master_url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data, session
    except Exception as e:
        print(f"Error retrieving master data: {e}")
        return None, session

def get_index_constituents(session, index_name):
    """
    Retrieve the constituents for a given index using the NSE API.
    The response JSON has a key "data" which is a list of constituents.
    The first element (index 0) is the index itself; skip it.
    Returns a list of constituent dictionaries.
    """
    encoded_index = urllib.parse.quote(index_name)
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={encoded_index}"
    try:
        response = session.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()
        all_constituents = data.get("data", [])
        # Skip the first element if its symbol matches the index name.
        if all_constituents and all_constituents[0].get("symbol", "").upper() == index_name.upper():
            constituents = all_constituents[0:]
        else:
            constituents = all_constituents
        return constituents
    except Exception as e:
        print(f"Error retrieving constituents for {index_name}: {e}")
        return []

def main():
    master_data, session = get_nse_master()
    if master_data is None:
        print("Failed to retrieve master data.")
        return
    # Filter for indices – try filtering by scripType "INDEX".
    indices = []
    for d in master_data.values():# if d.get("scripType", "").upper() == "INDEX"]
      indices.extend(d)

    # If that yields nothing, fall back to filtering for "NIFTY" in symbol.
    if not indices:
        indices = [d for d in master_data if "NIFTY" in d.get("symbol", "").upper()]

    print(f"Found {len(indices)} indices in NSE master data.")
    rows = []
    # Loop over each index with a progress bar.
    for idx_type, d in tqdm(master_data.items(), desc="Processing indices", ncols=100):
        for idx in d:
          index_name = idx
          index_type = idx_type  # Expected to be "INDEX"
          constituents = get_index_constituents(session, index_name)
          if constituents:
              for c in constituents:
                  # Create a row with the index info and constituent info.
                  row = {
                      "Index": index_name,
                      "Index Type": index_type,
                      "Stock": c.get("symbol", None)
                  }
                  # Include all additional columns from the constituent data.
                  for k, v in c.items():
                      if k not in row:
                          row[k] = v
                  rows.append(row)
    if rows:
        final_df = pd.DataFrame(rows)
        # Optionally, reorder columns so that "Index", "Index Type", and "Stock" come first.
        cols = final_df.columns.tolist()
        for col in ["Index", "Index Type", "Stock"]:
            if col in cols:
                cols.remove(col)
                cols.insert(0, col)
        final_df = final_df[cols]
    
        # Ensure data folder exists.
        if not os.path.exists("data"):
            os.makedirs("data")

        # Convert Index column to uppercase for consistent matching
        final_df["Index"] = final_df["Index"].str.upper()

        # Identify Mid Cap and Small Cap stocks
        midcap_stocks = set(final_df[final_df["Index"].str.contains(" MIDCAP ")]['Stock'])
        smallcap_stocks = set(final_df[final_df["Index"].str.contains(" SMALLCAP ")]['Stock'])

        # Function to classify companies
        def classify_market_cap(stock):
            if stock in midcap_stocks:
                return "Mid Cap"
            elif stock in smallcap_stocks:
                return "Small Cap"
            else:
                return "Large Cap"

        # Apply classification
        final_df["Market Cap Classification"] = final_df["Stock"].apply(classify_market_cap)
        
        # Save the final DataFrame to CSV.
        final_df.to_csv("data/index_constituents.csv", index=False)
        print("\nFinal Index Constituents Table saved to data/index_constituents.csv")
    else:
        print("No constituent data retrieved.")

if __name__ == "__main__":
  main()
