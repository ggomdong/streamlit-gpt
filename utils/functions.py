from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
import streamlit as st


def search_url_wikipedia(inputs):
    query = inputs["query"]
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)


def search_url_duckduckgo(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchRun()
    return ddg.run(query)


def load_website(inputs):
    url = inputs["url"]
    wl = WebBaseLoader([url])
    docs = wl.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text


def save_file(inputs):
    text = inputs["text"]
    with open('research_report.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    st.download_button(label="다운로드", file_name="research_report.txt", data=text)
    return "저장 완료"


functions_map = {
    "search_url_wikipedia": search_url_wikipedia,
    "search_url_duckduckgo": search_url_duckduckgo,
    "load_website": load_website,
    "save_file": save_file,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "search_url_wikipedia",
            "description": """
                질의를 인수로 받아, Wikipedia 검색 결과의 URL 링크 정보를 수집합니다.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "URL을 찾기위한 질의입니다. 예시 질의: Ransomware에 대해 조사해줘.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_url_duckduckgo",
            "description": """
                질의를 인수로 받아, DuckDuckGo 검색 결과의 URL 링크 정보를 수집합니다.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "URL을 찾기위한 질의입니다. 예시 질의: Ransomware에 대해 조사해줘.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_website",
            "description": """
                Wikipedia 와 DuckDuckGo에서 찾은 URL에서 text를 로드해서 문서로 만듭니다.
                문서의 언어는 한글이며, 문서의 마지막에 관련 URL을 제공합니다.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Wikipedia 와 DuckDuckGo에서 찾은 URL 입니다.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "load_website 함수를 통해 생성된 문서를 파일에 저장합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "load_website 함수를 통해 생성된 문서입니다",
                    },
                },
                "required": ["text"],
            },
        },
    },
]