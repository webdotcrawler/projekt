import os
import re
import json
import subprocess
from datetime import datetime
from bs4 import BeautifulSoup
from web_scraper import fetch_html_selenium, html_to_markdown, scrape_url
from data_source import URLS
from urllib.parse import urlparse, urljoin

def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]
    domain = re.sub(r'^www\.', '', domain)
    clean_domain = re.sub(r'\W+', '_', domain)
    return f"{clean_domain}_{timestamp}"

def extract_pagination_links(soup, base_url):
    """
    Extracts pagination URLs from a BeautifulSoup object.
    """
    pagination_urls = []
    pagination = soup.find("nav", class_="woocommerce-pagination")
    
    if pagination:
        for link in pagination.find_all("a", href=True):
            full_url = urljoin(base_url, link['href'])
            if full_url not in pagination_urls:
                pagination_urls.append(full_url)

    return pagination_urls

def scrape_multiple_urls(urls, fields, selected_model):
    output_folder = os.path.join('output', generate_unique_folder_name(urls[0]))
    os.makedirs(output_folder, exist_ok=True)
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0
    all_data = []
    first_url_markdown = None
    
    for url in urls:
        page_urls_to_scrape = [url]
        scraped_urls = set()  # Keep track of already scraped URLs
        
        while page_urls_to_scrape:
            current_url = page_urls_to_scrape.pop(0)
            if current_url in scraped_urls:
                continue
            
            raw_html = fetch_html_selenium(current_url)
            soup = BeautifulSoup(raw_html, "html.parser")
            markdown = html_to_markdown(raw_html)
            
            if first_url_markdown is None:
                first_url_markdown = markdown

            input_tokens, output_tokens, cost, formatted_data = scrape_url(
                current_url, fields, selected_model, output_folder, len(all_data) + 1, markdown)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            all_data.append(formatted_data)
            scraped_urls.add(current_url)

            pagination_urls = extract_pagination_links(soup, current_url)
            
            for page_url in pagination_urls:
                if page_url not in scraped_urls and page_url not in page_urls_to_scrape:
                    page_urls_to_scrape.append(page_url)
                    print(f"Added new page URL to scrape: {page_url}")

    return output_folder, total_input_tokens, total_output_tokens, total_cost, all_data, first_url_markdown

def perform_scrape():
    urls = URLS  # Assuming URLs is a list of URLs
    model_selection = "Groq Llama3.1 70b"  # Select the model you want to use
    tags = []  # You can populate this list as needed

    all_data = []
    output_folder, total_input_tokens, total_output_tokens, total_cost, all_data, first_url_markdown = scrape_multiple_urls(urls, tags, model_selection)

    return all_data, total_input_tokens, total_output_tokens, total_cost, output_folder

if __name__ == "__main__":
    results = perform_scrape()
    all_data, input_tokens, output_tokens, total_cost, output_folder = results

    try:
        # Use subprocess to trigger the combination of XLSX files
        combine_command = f"python merged_scraped_data.py {output_folder}"
        subprocess.check_call(combine_command, shell=True)
        print("Combining XLSX files completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while combining XLSX files: {e}")

    print("Scraping completed.")
    print(f"Total Input Tokens: {input_tokens}")
    print(f"Total Output Tokens: {output_tokens}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Results saved in folder: {output_folder}")
    for data in all_data:
        print(json.dumps(data, indent=4))
