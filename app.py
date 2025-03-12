import os
import asyncio
import streamlit as st
from src.scrapper.scrap import WebScraper
from src.embeddings.vector_store import VectorStore
from src.function.chatbot import process_query

os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("Botty 1.0 🤖")

# VectorStore.clear_database()
vec = VectorStore()

urls = st.text_area('Enter URLs (one per line)')
run_scraper = st.button('Run Scraper', disabled=not urls.strip())


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'scraping_done' not in st.session_state:
    st.session_state.scraping_done = False

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if run_scraper:
    st.write('Fetching and processing URLs... This may take a while..')

    scraper = WebScraper(urls.split('/n'))

    # ✅ Run the async function safely
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        split_docs = loop.run_until_complete(scraper.scrape())  # ✅ Fix async issue
    finally:
        loop.close()

    if not split_docs:
        st.error("No data extracted from the URLs.")

    else:
        st.session_state.vector_store = vec.initialize(split_docs)
        st.session_state.scraping_done = True
        st.success("Scraping and processing completed!")



# Display chat functionality
if not st.session_state.scraping_done:
    st.warning("Scrape some data first to enable chat!")
else:
    st.write("### Chat With Botty 💬")

    # Display chat history
    for message in st.session_state.messages:
        role, text = message["role"], message["text"]
        with st.chat_message(role):
            st.write(text)

    # User query input
    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.messages.append({"role": "user", "text": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        response, source_url = process_query(user_query, st.session_state.vector_store)

        formatted_response = f"**Answer:** {response}"
        if response != "I can't find your request in the provided context." and source_url:
            formatted_response += f"\n\n**Source:** {source_url}"

        # Append response
        st.session_state.messages.append({"role": "assistant", "text": formatted_response})
        with st.chat_message("assistant"):
            st.write(formatted_response)