import logging
import time
from pathlib import Path
from typing import List
import requests
import pickle
import uuid
import argparse
import os

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CHROME_DRIVER_VERSION = "125.0.6422.7"

class URLObject:
    def __init__(self, id: str, url: str):
        self.id = id
        self.url = url


class CrawledURLObject(URLObject):
    def __init__(
        self,
        id: str,
        url: str,
        content: str,
        metadata={},
        is_chunk=False,
        is_error=False,
        error_message=None,
    ):
        super().__init__(id, url)
        self.content = content
        self.metadata = metadata
        self.is_chunk = is_chunk
        self.is_error = is_error
        self.error_message = error_message


class Loader:
    def __init__(self):
        pass

    def to_crawled_obj(self, url_list: List[str]):    
        url_objs = [URLObject(str(uuid.uuid4()), url) for url in url_list]
        crawled_url_objs = self.crawl_urls(url_objs)
        return crawled_url_objs

    def crawl_urls(self, url_objects: list[URLObject]) -> List[CrawledURLObject]:
        logger.info(f"Start crawling {len(url_objects)} urls")
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-pipe")
        chrome_driver_path = Path(ChromeDriverManager(driver_version=CHROME_DRIVER_VERSION).install())
        options.binary_location = str(chrome_driver_path.parent.absolute())
        logger.info(f"chrome binary location: {options.binary_location}")
        driver = webdriver.Chrome(options=options)

        docs: List[CrawledURLObject] = []
        for url_obj in url_objects:
            try:
                logger.info(f"loading url: {url_obj.url}")
                driver.get(url_obj.url)
                time.sleep(2)
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                text_list = []
                for string in soup.strings:        
                    if string.find_parent("a"):
                        href = urljoin(url_obj.url, string.find_parent("a").get("href"))
                        if href.startswith(url_obj.url):
                            text = f"{string} {href}"
                            text_list.append(text)
                    elif string.strip():
                        text_list.append(string)
                text_output = "\n".join(text_list)
                
                title = url_obj.url
                for title in soup.find_all("title"):
                    title = title.get_text()
                    break

                docs.append(
                    CrawledURLObject(
                        id=url_obj.id,
                        url=url_obj.url,
                        content=text_output,
                        metadata={"title": title, "source": url_obj.url},
                    )
                )

            except Exception as err:
                logger.info(f"error crawling {url_obj}")
                logger.error(err)
                docs.append(
                    CrawledURLObject(
                        id=url_obj.id,
                        url=url_obj.url,
                        content=None,
                        metadata={"title": url_obj.url, "source": url_obj.url},
                        is_error=True,
                        error_message=str(err),
                    )
                )
        driver.quit()
        return docs

    def get_all_urls(self, base_url: str, max_num: int) -> List[str]:
        logger.info(f"Getting all pages for base url: {base_url}, maximum number is: {max_num}")
        urls_visited = []
        base_url = base_url.split("#")[0].rstrip("/")
        urls_to_visit = [base_url]
        
        while urls_to_visit:
            if len(urls_visited) >= max_num:
                break
            current_url = urls_to_visit.pop(0)
            if current_url not in urls_visited:
                urls_visited.append(current_url)
                new_urls = self.get_outsource_urls(current_url, base_url)
                urls_to_visit.extend(new_urls)
                urls_to_visit = list(set(urls_to_visit))
        logger.info(f"URLs visited: {urls_visited}")
        return sorted(urls_visited[:max_num])
    
    def get_outsource_urls(self, curr_url: str, base_url: str):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'
        } 
        new_urls = list()
        try:
            response = requests.get(curr_url, headers=headers, timeout=10)
            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all("a"):
                    try:
                        full_url = urljoin(curr_url, link.get("href"))
                        full_url = full_url.split("#")[0].rstrip("/")
                        if self._check_url(full_url, base_url):
                            new_urls.append(full_url)
                    except Exception as err:
                        logger.error(f"Fail to process sub-url {link.get('href')}: {err}")
            else:
                logger.error(f"Failed to retrieve page {curr_url}, status code: {response.status_code}")
        except Exception as err:
            logger.error(f"Fail to get the page from {curr_url}: {err}")
        return list(set(new_urls))
    
    def _check_url(self, full_url, base_url):
        kw_list = ['.pdf', '.jpg', '.png', '.docx', '.xlsx', '.pptx', '.zip', ".jpeg"]
        if full_url.startswith(base_url) and full_url and not any(kw in full_url for kw in kw_list) and full_url != base_url:
            return True
        return False

    def get_candidates_websites(self, urls: List[CrawledURLObject], top_k: int) -> List[str]:
        """Based on the pagerank algorithm of the crawled websites, return the top 10 websites. 
        The reason why we can do that is because we have the hreqs of the including <a> tags in the content of the website.
        So we can use that to construct the edges and then use the tool from networkx to get the pagerank of the websites.
        """
        
        nodes = []
        edges = []
        url_to_id_mapping = {}
        for url in urls:
            url_to_id_mapping[url.url] = url.id

        for url in urls:
            if url.is_error:
                continue
            for url_key in url_to_id_mapping:
                if url_key in url.content:
                    edge = [url.id, url_to_id_mapping[url_key]]
                    edges.append(edge)

            node = [url.id, {"url": url.url, "content": url.content, "metadata": url.metadata}]
            nodes.append(node)
        
        self.graph = nx.DiGraph(name="website graph")
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        pr = nx.pagerank(self.graph, alpha=0.9)
        # sort the pagerank values in descending order
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"pagerank results: {sorted_pr}")
        # get the top websites
        top_k_websites = sorted_pr[:top_k]
        urls_candidates = [self.graph.nodes[url_id] for url_id, _ in top_k_websites]
        urls_cleaned = [doc for doc in urls_candidates if doc]
        return urls_cleaned
    
    @staticmethod
    def save(file_path: str, docs: List[CrawledURLObject]):
        with open(file_path, "wb") as f:
            pickle.dump(docs, f)
    
    @classmethod
    def chunk(cls, url_objs: List[CrawledURLObject]) -> List[CrawledURLObject]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=40)
        docs = []
        langchain_docs = []
        for url_obj in url_objs:
            if url_obj.is_error or url_obj.content is None:
                logger.info(f"Skip url: {url_obj.url} because of error or no content")
                continue
            elif url_obj.is_chunk:
                logger.info(f"Skip url: {url_obj.url} because it has been chunked")
                docs.append(url_obj)
                continue
            splitted_text = text_splitter.split_text(url_obj.content)
            for i, txt in enumerate(splitted_text):
                doc = CrawledURLObject(
                    id=url_obj.id+"_"+str(i),
                    url=url_obj.url,
                    content=txt,
                    metadata=url_obj.metadata,
                    is_chunk=True,
                )
                docs.append(doc)
                langchain_docs.append(Document(page_content=txt, metadata={"source": url_obj.url}))
        return langchain_docs

