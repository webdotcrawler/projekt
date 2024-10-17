import sys
import os
import pandas as pd
import json
import time
import re
from groq import Groq
from dotenv import load_dotenv
from cachetools import TTLCache
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Load LLM settings
LLAMA_MODEL_FULLNAME = os.environ.get("LLAMA_MODEL_FULLNAME", "llama-3.1-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Cache to store previous LLM column responses (TTL: 24 hours)
column_cache = TTLCache(maxsize=100, ttl=86400)

# Global usage trackers
requests_in_current_minute = 0
tokens_in_current_minute = 0
tokens_in_current_day = 0

# Rate limit constants
RATE_LIMITS = {
    "requests_per_minute": 30,
    "tokens_per_minute": 20000,
    "tokens_per_day": 500000
}

# Initialize SentenceTransformer for embedding-based similarity
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Rate limit error handler
def handle_rate_limit_error(error_message):
    wait_time_match = re.search(r"Please try again in ([\d\.]+)s", error_message)
    if wait_time_match:
        wait_seconds = float(wait_time_match.group(1))
        print(f"Rate limit reached. Waiting for {wait_seconds} seconds...")
        return wait_seconds
    else:
        print("Failed to parse wait time from error. Using exponential backoff.")
        return None

# Extract JSON from LLM response
def extract_json(response_content):
    json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    print("No valid JSON found in the response.")
    return None

# LLM request with backoff
def request_with_backoff(client, messages):
    global requests_in_current_minute, tokens_in_current_minute, tokens_in_current_day
    retry_count = 0
    max_retries = 10
    wait_time = 1

    token_count = sum(len(message['content'].split()) for message in messages)

    if tokens_in_current_minute + token_count > RATE_LIMITS["tokens_per_minute"]:
        time_to_wait = 60
        print(f"Token limit reached. Waiting {time_to_wait} seconds...")
        time.sleep(time_to_wait)
        tokens_in_current_minute = 0

    if tokens_in_current_day + token_count > RATE_LIMITS["tokens_per_day"]:
        print("Daily token limit exceeded. Try again tomorrow.")
        return None

    while retry_count < max_retries:
        try:
            if requests_in_current_minute >= RATE_LIMITS["requests_per_minute"]:
                time_to_wait = 180
                print(f"Request limit reached. Waiting {time_to_wait} seconds...")
                time.sleep(time_to_wait)
                requests_in_current_minute = 0

            # Send request to Groq API
            completion = client.chat.completions.create(
                model=LLAMA_MODEL_FULLNAME,
                messages=messages
            )

            tokens_in_current_minute += token_count
            tokens_in_current_day += token_count
            requests_in_current_minute += 1

            # Process LLM response
            response_content = completion.choices[0].message.content
            print("LLM response content:", response_content)

            json_response = extract_json(response_content)
            if json_response:
                return json.loads(json_response)

            print("No valid JSON found in the response. Retrying...")
            retry_count += 1
            time.sleep(wait_time)
            wait_time *= 2

        except Exception as e:
            print(f"Request failed: {e}")
            if "rate_limit_exceeded" in str(e):
                wait_seconds = handle_rate_limit_error(str(e))
                if wait_seconds:
                    time.sleep(wait_seconds)
                    retry_count = 0
                else:
                    time.sleep(wait_time)
                    wait_time *= 2
                    retry_count += 1
            elif "503" in str(e):
                print("Service Unavailable (503). Retrying...")
                time.sleep(wait_time)
                wait_time *= 2
                retry_count += 1
            else:
                print("Non-rate-limit error occurred. Exiting retry loop.")
                break

    print("Max retries reached. Exiting.")
    return None

# Flatten listings from JSON
def flatten_listings(data):
    flattened_data = []
    for entry in data:
        if isinstance(entry, dict) and 'listings' in entry:
            if isinstance(entry['listings'], list):
                for listing in entry['listings']:
                    flattened_data.append(listing)
            else:
                print(f"Warning: 'listings' is not a list in {entry}.")
        else:
            print(f"Warning: 'listings' key not found in {entry}.")
    return flattened_data

# Fetch column mappings from LLM
def fetch_columns_from_llm(data_sample):
    if not data_sample.strip():
        print("No column names provided to LLM.")
        return {}

    client = Groq(api_key=GROQ_API_KEY)
    system_message = """
    You are an intelligent assistant that helps clean and organize data.
    Given these column names, please suggest merged or unified columns for consistency.
    Provide a unified column mapping and return ONLY the column names as a dictionary in JSON format.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": data_sample}
    ]

    unified_columns_response = request_with_backoff(client, messages)
    if unified_columns_response and isinstance(unified_columns_response, dict):
        return unified_columns_response

    print(f"Unexpected LLM response format: {unified_columns_response}")
    return {}

# String similarity function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Embedding similarity function
def get_embedding_similarity(a, b):
    embeddings = embedding_model.encode([a, b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Column unification logic
def unify_columns(data):
    """
    Use LLM to suggest unified column names based on the data provided.
    Additionally, use both string similarity and embedding similarity to merge columns.
    """
    # Extract sample data to send to LLM
    column_samples = {col: data[col].head(3).tolist() for col in data.columns if col in data}
    
    # Convert column samples to JSON and pass to LLM for suggestion
    data_sample = json.dumps(column_samples)
    unified_columns_map = fetch_columns_from_llm(data_sample)

    if unified_columns_map:
        print(f"Unified columns mapping received: {unified_columns_map}")

        for new_col, old_cols in unified_columns_map.items():
            if isinstance(old_cols, str):
                old_cols = [old_cols]  # Ensure single entries are treated as a list
            else:
                old_cols = old_cols.split(', ')  # Split multiple columns if provided as comma-separated string

            existing_data = []
            for col in old_cols:
                if col in data.columns:
                    existing_data.append(data[col].dropna())  # Drop NaN values

            if existing_data:
                data[new_col] = pd.concat(existing_data, ignore_index=True)  # Merge all existing data into the new column

            data.drop(columns=old_cols, errors='ignore', inplace=True)

    # Handle columns with similar names using both string and embedding similarity
    for col in data.columns:
        for other_col in data.columns:
            if col != other_col:
                string_sim = similar(col.lower(), other_col.lower())
                embedding_sim = get_embedding_similarity(col.lower(), other_col.lower())

                if string_sim > 0.8 or embedding_sim > 0.8:
                    print(f"Columns '{col}' and '{other_col}' are similar (string_sim={string_sim:.2f}, embedding_sim={embedding_sim:.2f}), merging...")
                    data[col] = data[col].combine_first(data[other_col])  # Merge non-NaN values
                    data.drop(columns=[other_col], inplace=True)

    return data

# Data cleaning and standardization
def clean_and_standardize(data):
    if 'price' in data.columns:
        data['price'] = data['price'].replace({'$': '', '£': '', '€': ''}, regex=True).astype(float)
        data['price'] = data['price'].apply(lambda x: f"${x:.2f}")
    
    for url_field in ['image_url', 'product_url']:
        if url_field in data.columns:
            data[url_field] = data[url_field].str.strip()
            data[url_field] = data[url_field].replace(r'http://', 'https://', regex=True)
    
    return data

# Combine JSON files into one DataFrame
def combine_json_files_first_pass(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.startswith("sorted_data") and f.endswith(".json")]
    all_data = []
    
    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), 'r') as f:
            try:
                data = json.load(f)
                if not data:
                    print(f"Warning: JSON file {json_file} is empty or not properly structured.")
                else:
                    all_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {json_file}: {e}")

    if all_data:
        flattened_data = pd.json_normalize(flatten_listings(all_data))
        print(f"Combined data preview:\n{flattened_data.head()}\n")
        return flattened_data
    else:
        print("No valid JSON data found.")
        return pd.DataFrame()  # Return empty DataFrame if no valid data found

# Save combined data to xlsx
def save_to_xlsx(data, output_path):
    data.to_excel(output_path, index=False)

# Main function
def main(folder_path, output_filename):
    combined_data = combine_json_files_first_pass(folder_path)
    
    if combined_data is not None and not combined_data.empty:
        print(f"Combined data columns before unification: {combined_data.columns.tolist()}")
        
        # Clean and standardize data
        combined_data = clean_and_standardize(combined_data)
        
        # Extract column samples for LLM
        column_samples = {col: combined_data[col].head(3).tolist() for col in combined_data.columns}
        data_sample = json.dumps(column_samples)
        
        # Unify columns
        unified_columns_map = fetch_columns_from_llm(data_sample)
        if unified_columns_map:
            print(f"Unified columns mapping received: {unified_columns_map}")
            combined_data = unify_columns(combined_data)
        
        print(f"Combined data columns after unification: {combined_data.columns.tolist()}")
        
        # Save to xlsx
        output_path = os.path.join(folder_path, output_filename)
        save_to_xlsx(combined_data, output_path)
    else:
        print("No valid JSON data found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merged_scraped_data.py <folder_path> <output_filename>")
    else:
        main(sys.argv[1], sys.argv[2])
