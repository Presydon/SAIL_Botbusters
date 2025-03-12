from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import chain


class WebScraper:
    def __init__(self, urls, persist_directory='/chromadb'):
        """ Initializes the web scraper with given URLs and persistence directory.
        :param urls: List of URLs to scrape.
        :param persist_directory: Directory to store processed data (optional).
         """
        self.urls = urls
        self.persist_directorry = persist_directory
        self.loader = AsyncChromiumLoader(urls=self.urls)
        self.text_transformer = Html2TextTransformer()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

    async def scrape(self):
        """ Asynchronously scrapes the provided URLs, extracts text, and splits it into chunks.
        :return: List of processed document chunks with metadata.
         """
        docs = await self.loader.aload()

        transformed_docs = self.text_transformer.transform_documents(docs)  

        split_docs_nested = [self.text_splitter.split_documents([doc]) for doc in transformed_docs]

        processed_chunks = []
        for doc_list, original_doc in zip(split_docs_nested, transformed_docs):
            for chunk in doc_list:
                chunk.metadata['source'] = original_doc.metadata.get('source', "Unknown")
                processed_chunks.append(chunk)

        return processed_chunks