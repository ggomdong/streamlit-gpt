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
import requests, os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def is_valid(key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False


# cache_data 사용시 UnserializableReturnValueError 가 발생하여 변경
@st.cache_resource(show_spinner="파일 임베딩 중...")
def embed_file(file, key):
    # .cache 폴더가 없으면 생성해준다.
    file_folder = './.cache/files'
    
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    file_content = file.read()
    file_path = f"{file_folder}/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            당신은 주어진 문서를 빠르게 탐색해서 질문에 답할 수 있습니다. 주어진 문서대로만 답하고, 꾸며내지 마세요.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("📃DocumentGPT")

st.markdown(
    """
문서의 내용을 질문해주세요. AI가 답변해 드립니다.
"""
)

with st.sidebar:
    st.markdown("""
    Github Repo : https://github.com/ggomdong/streamlit-gpt
    """)

    key = st.text_input("OPEN_API_KEY", placeholder="OPENAI_API_KEY를 입력해주세요.", type="password")

    if key:
        # OPENAI_API_KEY 가 입력되면 파일 업로드 가능
        if is_valid(key):
            st.success("유효한 OPENAI_API_KEY 입니다.")

            try:
                file = st.file_uploader(
                    "TXT, PDF, DOCX 확장자를 가진 파일을 업로드하세요.",
                    type=["pdf", "txt", "docx"],
                )
            except:
                st.error("파일 업로드에 실패하였습니다.")
        else:
            st.warning("올바른 OPENAI_API_KEY를 입력하세요.")
            key = ""

if key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini-2024-07-18",
        streaming=True,
        callbacks=[ChatCallbackHandler(),],
        api_key=key
    )

    if file:
        retriever = embed_file(file, key)
        send_message("반갑습니다! 질문해 주세요. ^^", "ai", save=False)
        paint_history()
        message = st.chat_input("문서에 대해 질문해 주세요.")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                } | prompt | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)
    else:
        st.session_state["messages"] = []