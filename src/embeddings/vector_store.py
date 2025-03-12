import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, persist_directory=os.path.abspath('./chromadb'), model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """ Initializes the vector store with an embedding model and a persistence directory.
        :param persist_directory: Directory where vector data is stored.
        :param model_name: Pretrained model for embeddings.
         """
        
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = None

    def initialize(self, split_docs):
        """ Initializes the Chroma vector store with provided documents.
        :param split_docs: List of document chunks for vectorsization.
        :return: Chroma vector store instance.
         """
        
        self.vector_db = Chroma.from_documents(
            documents = split_docs,
            embedding = self.embeddings,
            persist_directory = self.persist_directory
        )
        
        return self.vector_db

    @staticmethod
    def clear_database():
        """ Clears the Chroma vector store if it exists """
        persist_directory = os.path.abspath('./chromadb')
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
                print(f'Deleted ChromaDB directory: {persist_directory}')
            except Exception as e:
                print(f"Error: {e}")

        else:
            print("No ChromaDB found to clear")
        # persist_directory = os.path.abspath('./chromadb')
        # if os.path.exists(persist_directory):
        #     try:
        #         shutil.rmtree(persist_directory)
        #         print('ChromaDB cleared')

        #     except Exception as e:
        #         print(f"Error: {e}")
        
        # else:
        #     print('No ChromaDB found to clear.')