# Import required libraries
import os
import streamlit as st
import validators
from dotenv import load_dotenv

# LangChain and Groq related imports
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Step 1: Load environment variables from .env file
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Step 2: Set up the Streamlit web app
st.set_page_config(page_title="Summarizer App", page_icon="📝")
st.title("📝 LangChain: Summarize Text from Website")
st.subheader("Enter a Website URL to get a short summary.")

# Step 3: Input field for the URL
input_url = st.text_input("Enter URL here", label_visibility="collapsed")

# Step 4: Initialize the Groq LLM with Gemma model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=os.getenv("GROQ_API_KEY"))

# Step 5: Create a prompt template for summarization
prompt_template = """
Provide a summary of the following content in approximately 300 words:

Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Step 6: Define summarization logic when the button is clicked
if st.button("Summarize the Content"):
    if not validators.url(input_url):
        st.error("❌ Please enter a valid URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing content..."):
                    loader = UnstructuredURLLoader(
                        urls=[input_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()

                    # Step 6.2: Summarize the content using the chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)

                    # Step 6.3: Display the summarized output
                    st.success("✅ Summary:")
                    st.write(summary)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")