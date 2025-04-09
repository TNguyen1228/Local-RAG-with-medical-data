import requests
from bs4 import BeautifulSoup
import csv

def crawl_links_with_class(url, class_name):
    # Send HTTP request to the website
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links with the specified class
        links = soup.find_all('a', class_=class_name)
        
        # Extract href and text from each link
        results = []
        for link in links:
            href = link.get('href')
            text = link.get_text(strip=True)
            results.append({'url': href, 'text': text})
            
        return results
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

def save_to_csv(data, filename='links.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Data saved to {filename}")

# Usage example
if __name__ == "__main__":
    # Replace with the actual URL of the website you want to crawl
    website_url = "https://tamanhhospital.vn/benh-hoc-a-z/"
    class_name = "cl_33"
    
    links = crawl_links_with_class(website_url, class_name)
    
    # Print the results
    for i, link in enumerate(links, 1):
        print(f"{i}. {link['text']} - {link['url']}")
    
    # Save to CSV
    if links:
        save_to_csv(links)