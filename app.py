from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import requests
import os

st.set_page_config(
    page_title="ggomdong's GPT Series",
    page_icon="📃",
)

st.title("ggomdong's GPT Series")

st.markdown(
    """
    GPT를 활용한 포트폴리오입니다.

    - [X] [DocumentGPT](/DocumentGPT) : 문서의 내용에 대해 대화합니다.
    - [] [PrivateGPT](/PrivateGPT) : 문서의 내용에 대해 대화합니다. (무료 버전)
    - [X] [QuizGPT](/QuizGPT) : 문서 혹은 위키피디아를 통해 퀴즈를 냅니다.
    - [] [SiteGPT](/SiteGPT) : 웹사이트의 내용에 대해 대화합니다.
    - [] [MeetingGPT](/MeetingGPT) : 동영상 혹은 음원의 내용에 대해 대화합니다.
    - [] [InvestorGPT](/InvestorGPT)
    """
)
