import csv
import os
import time
import random
import logging
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

def read_links_from_csv(filename='links.csv'):
    links = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                links.append(row)
    return links

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--enable-unsafe-swiftshader")

    
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def extract_content_with_selenium(url, driver, max_retries=3):
    for retry in range(max_retries):
        try:
            logging.info(f"Attempting to fetch {url} with Selenium (Retry {retry+1}/{max_retries})")
            driver.get(url)
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.ID, "ftwp-postcontent"))
                )
            except TimeoutException:
                logging.warning(f"Timeout waiting for content div in {url}")
                if retry == max_retries - 1:
                    return None
                continue
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            content_div = soup.find('div', id='ftwp-postcontent')
            if not content_div:
                logging.warning(f"No 'ftwp-postcontent' div found in {url}")
                return None
            for unwanted in content_div.find_all('div', class_=['ftwp-in-post ftwp-float-center', 'content_insert']):
                unwanted.decompose()
            h2_tags = content_div.find_all('h2')
            if not h2_tags:
                logging.warning(f"No h2 tags found in {url}")
                return None
            result = []
            start_extracting = False
            current_section = []
            for element in content_div.children:
                if element.name == 'figure':
                    continue
                if not element.name and not element.string:
                    continue
                if element.name == 'h2' and not start_extracting:
                    start_extracting = True
                if start_extracting:
                    if element.name == 'h2' and current_section:
                        result.append('\n'.join(current_section))
                        result.append("SEPARATED")
                        current_section = []
                    if element.name == 'ul':
                        for li in element.find_all('li'):
                            text = li.get_text(strip=True)
                            if text:
                                current_section.append(text)
                                current_section.append('')
                    elif element.name == 'p':
                        text = ' '.join(element.stripped_strings)
                        if text:
                            current_section.append(text)

                    elif element.name and element.name != 'ul':
                        text = element.get_text(strip=True)
                        if text:
                            current_section.append(text)
            if current_section:
                result.append('\n'.join(current_section))
            return '\n'.join(result)
        except WebDriverException as e:
            logging.error(f"Selenium error on URL {url}: {e}")
            if retry == max_retries - 1:
                return None
            wait_time = (2 ** retry) * random.uniform(2, 4)
            logging.info(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
            try:
                driver.quit()
                driver = setup_driver()
            except:
                pass
    return None

def get_filename_from_url(url):
    # Extract the path from the URL
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path.strip('/')
    
    # Get the last part of the path (the slug)
    if path:
        filename = path.split('/')[-1]
        # Remove trailing slashes if any
        filename = filename.rstrip('/')
        # Add .txt extension
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        return filename
    else:
        # If no path, use the domain as filename
        return f"{parsed_url.netloc}.txt"

def get_processed_files(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return set()
    
    return {file for file in os.listdir(output_dir) if file.endswith('.txt') and not file.startswith('_failed_')}

def process_all_links(output_dir='extracted_content', min_delay=2, max_delay=5, start_index=0, end_index=None):
    links = read_links_from_csv()
    
    # Limit the range if specified
    if end_index is None:
        end_index = len(links)
    links_to_process = links[start_index:end_index]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of already processed files
    processed_files = get_processed_files(output_dir)
    logging.info(f"Found {len(processed_files)} already processed files")
    
    # Setup Selenium driver
    driver = setup_driver()
    
    # Keep track of failed URLs
    failed_urls = []
    
    try:
        for i, link in enumerate(links_to_process, start_index + 1):
            url = link['url']
            title = link['text']
            
            # Generate filename from URL
            filename = get_filename_from_url(url)
            
            # Skip if already processed
            if filename in processed_files:
                logging.info(f"Skipping {i}/{len(links)}: {title} (already processed)")
                continue
            
            logging.info(f"Processing {i}/{len(links)}: {title}")
            
            # Extract content using Selenium
            content = extract_content_with_selenium(url, driver)
            
            if content:
                # Save to individual file
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Add URL and title as header
                    f.write(f"URL: {url}\n")
                    f.write(f"TITLE: {title}\n")
                    f.write("=" * 50 + "\n")
                    f.write(content + "\n")
                
                logging.info(f"Successfully saved content to {file_path}")
            else:
                logging.error(f"Failed to extract content from {url}")
                failed_urls.append({'url': url, 'title': title})
                
                # Write to a "failed" file to keep track
                with open(os.path.join(output_dir, f"_failed_{filename}"), 'w', encoding='utf-8') as f:
                    f.write(f"URL: {url}\n")
                    f.write(f"TITLE: {title}\n")
                    f.write("=" * 50 + "\n")
                    f.write("Failed to extract content\n")
            
            # Add a variable delay to avoid detection
            delay = random.uniform(min_delay, max_delay)
            logging.info(f"Waiting {delay:.2f} seconds before next request")
            time.sleep(delay)
            
    finally:
        # Close the driver when done
        try:
            driver.quit()
        except:
            pass
        
        # Write summary of failed URLs
        if failed_urls:
            with open(os.path.join(output_dir, "_failed_urls.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Total failed URLs: {len(failed_urls)}\n\n")
                for item in failed_urls:
                    f.write(f"URL: {item['url']}\n")
                    f.write(f"TITLE: {item['title']}\n")
                    f.write("-" * 50 + "\n")
                    
            logging.info(f"Content extraction complete. {len(failed_urls)} URLs failed, check {output_dir}/_failed_urls.txt for details")
        else:
            logging.info(f"Content extraction complete. All URLs processed successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract content from URLs in a CSV file using Selenium')
    parser.add_argument('--min-delay', type=float, default=5, help='Minimum delay between requests in seconds')
    parser.add_argument('--max-delay', type=float, default=10, help='Maximum delay between requests in seconds')
    parser.add_argument('--output', type=str, default='extracted_content', help='Output directory name')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive, None for all)')
    
    args = parser.parse_args()
    
    process_all_links(
        output_dir=args.output, 
        min_delay=args.min_delay, 
        max_delay=args.max_delay,
        start_index=args.start,
        end_index=args.end
    )