__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

# Custom Web Scraper Tool
class WebScraperTool(BaseTool):
    name: str = "WebScraper"  
    description: str = "Scrapes and retrieves the main content from a given webpage URL."  

    def _run(self, url: str) -> str:  # Added return type annotation
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            # Get visible text only
            paragraphs = soup.find_all("p")
            content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return content  # Limit content length for processing
        except Exception as e:
            return f"Error retrieving content: {str(e)}"

# Streamlit UI
st.title("🧠 AI Web Content Summarizer")

with st.form("url_form"):
    url = st.text_input("Enter a Webpage URL to Analyze", placeholder="https://example.com/some-article")
    submit = st.form_submit_button("Summarize")

if submit and url.strip():
    with st.spinner("Processing..."):

        # Define Agents
        content_retriever = Agent(
            role="Web Content Retriever",
            goal="Retrieve and clean content from the given URL.",
            backstory="You're skilled at extracting readable information from HTML pages.",
            tools=[WebScraperTool()],
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

        info_extractor = Agent(
            role="Key Information Extractor",
            goal="Extract important points and data from the web content.",
            backstory="You're trained to filter out noise and find what's most relevant in a text.",
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

        summarizer = Agent(
            role="Content Summarizer",
            goal="Generate a concise, clear, and informative summary from the key points.",
            backstory="You are a professional writer known for summarizing complex articles into digestible summaries.",
            llm=llm,
            allow_delegation=False,
            verbose=True
        )

        # Define Tasks
        retrieve_task = Task(
            description=f"Retrieve the full readable content from this URL: {url}",
            expected_output="Clean, readable text extracted from the web page.",
            agent=content_retriever
        )

        extract_task = Task(
            description="Extract the most important information, key facts, and insights from the web content.",
            expected_output="Bullet points or JSON with key highlights.",
            context=[retrieve_task],
            agent=info_extractor
        )

        summarize_task = Task(
            description="Write a detailed, structured summary of the extracted key information from the webpage.",
            expected_output="A clear and concise summary with heading and subheading.",
            context=[extract_task],
            agent=summarizer
        )

        # Execute the crew
        crew = Crew(
            agents=[content_retriever, info_extractor, summarizer],
            tasks=[retrieve_task, extract_task, summarize_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()

        # Display result
        st.header("🔍 Summary of the Webpage")
        st.markdown(result)

