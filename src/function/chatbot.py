import os
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from conversation.talks import SmallTalkManager
from dotenv import load_dotenv

load_dotenv()

stm = SmallTalkManager()

groq_api = os.getenv('GROQ_API_KEY')

LLM = ChatGroq(model='llama-3.2-1b-preview', groq_api_key=groq_api, temperature=0)

system_prompt = """
    You are Botty, an AI assistant that answers questions **strictly based on the retrieved context**.

    - **If the answer is found in the context, respond concisely.**  
    - **If the answer is NOT in the context, reply ONLY with: "I can't find your request in the provided context."**   
    - **If the question is unrelated to the provided context, reply ONLY with: "I can't answer that."**  
    - **DO NOT use external knowledge or assumptions.**  

    Context:  
    {context}  

    Now, respond accordingly
    """

PROMPT = ChatPromptTemplate([("system", system_prompt), ("human", "{input}")])


if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def run_asyncio_coroutine(coro):
    """Helper function to run async functions in a blocking manner."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def process_query(user_query, vector_store):
    """Processes the user query and returns an AI-generated response."""
    user_query_cleaned = stm.clean_input(user_query)
    response = ""
    source_url = ""


    if user_query_cleaned in stm.load_small_talks():
        small_talks = stm.load_small_talks()  
        response = small_talks[user_query_cleaned]  
        source_url = 'Knowledge base'

    else:
        retriever = vector_store.as_retriever(search_kwargs={'k': 7})
        scraper_chain = create_stuff_documents_chain(llm= LLM, prompt=PROMPT)
        llm_chain = create_retrieval_chain(retriever, scraper_chain)

        retrieved_docs = retriever.invoke(user_query_cleaned)

        # retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        # scraper_chain = create_stuff_documents_chain(llm=LLM, prompt=PROMPT)
        # llm_chain = create_retrieval_chain(retriever, scraper_chain)

        # retrieved_docs = retriever.invoke(user_query_cleaned)

        if retrieved_docs:
            response = llm_chain.invoke({'input': user_query_cleaned})['answer']
            source_url = retrieved_docs[0].metadata.get('source', "Unknown")

            if not response.strip():
                response = "I can't find your request in the provided context."
                source_url = "No source found"

        else:
            response = "I can't find your request in the provided context."
            source_url = ""

    return response, source_url


