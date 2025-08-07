from langchain.tools import Tool
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def web_scraper(url: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] WebScraper invoked with URL: {url}
          \n-------------------------------------------------\n""")
    
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    page_text = soup.get_text(separator=' ', strip=True)

    # Find all tables
    tables = soup.find_all('table')

    tables_data = []

    for table in soup.find_all("table"):
        caption_tag = table.find("caption")
        if caption_tag:
            caption = caption_tag.get_text(strip=True)
            df = pd.read_html(str(table))[0]
            markdown = df.to_markdown(index=False)
            tables_data.append(f"\n### {caption} ###\n{markdown}")

    tables_output = "\n\n".join(tables_data)
    
    output = f"--- PAGE TEXT (first 1000 chars) ---\n{page_text[:1000]}\n\n"
    output += f"--- EXTRACTED TABLES ---\n{tables_output if tables_output else 'No readable tables found.'}"
    
    return output[:10000]

web_scraper_tool = Tool(
    name="WebScraper",
    func=web_scraper,
    description="Use this to scrape text content from a given URL. Input should be a URL."
)
