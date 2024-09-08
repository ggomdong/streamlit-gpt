import json, requests, os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

function = {
    "name": "create_quiz",
    "description": "문답 세트를 받아서 퀴즈를 반환하는 함수",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}



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


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="파일 로딩 중...")
def split_file(file):
    # .cache 폴더가 없으면 생성해준다.
    file_folder = './.cache/files'
    
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    file_content = file.read()
    file_path = f"{file_folder}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


# streamlit의 캐시 함수에 변경 가능(mutable)한 parameter를 전달하면 해시할 수 없어서 오류가 발생함
# docs는 list이고 변경 가능하므로 해시 불가하므로, _를 붙여줘서 오류 방지
# 캐시 함수에 활용할 parameter가 필요하므로, topic(여기서는 검색어 혹은 파일명)을 추가해줘서 캐시 기능을 완성
@st.cache_data(show_spinner="퀴즈 생성 중...")
def run_quiz_chain(_docs, topic, total_count, difficulty):
    prompt = PromptTemplate.from_template("{context}에 대한 퀴즈를 {total_count}개 만들어줘. 한글로 작성해줘. 난이도는 {difficulty}")
    # chain = {"context": _docs | RunnableLambda(format_docs)} | prompt | llm
    chain = prompt | llm
    # return chain.invoke(_docs)
    return chain.invoke({"context": _docs, "total_count": total_count,  "difficulty": difficulty})


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("❓QuizGPT")

with st.sidebar:
    st.markdown("""
    Github Repo : https://github.com/ggomdong/streamlit-gpt
    """)
    # 변수 초기화
    # difficulty: 난이도, total_count: 퀴즈 갯수, current: 현재퀴즈번호, count: 정답수
    docs = None
    topic = None
    difficulty = None
    total_count = 0
    current = 0
    count = 0
    button = None

    key = st.text_input("OPEN_API_KEY", placeholder="OPENAI_API_KEY를 입력해주세요.", type="password")

    if key:
        # OPENAI_API_KEY 가 입력되면 파일 업로드 가능
        if is_valid(key):
            st.success("유효한 OPENAI_API_KEY 입니다.")

            llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-4o-mini-2024-07-18",
                api_key=key
            ).bind(
                function_call={
                    "name": "create_quiz",
                },
                functions=[
                    function,
                ],
            )

            difficulty = st.selectbox(
                "퀴즈의 난이도를 선택해 주세요.",
                (
                    "쉬움",
                    "어려움",
                ),
            )
            total_count = st.number_input(
                "퀴즈의 갯수를 선택해 주세요.(3~10개)", min_value=3, max_value=10, value="min"
            )
            choice = st.selectbox(
                "무엇으로부터 퀴즈를 만들까요?",
                (
                    "File",
                    "Wikipedia",
                ),
            )
            if choice == "File":
                try:
                    file = st.file_uploader(
                        "TXT, PDF, DOCX 확장자를 가진 파일을 업로드하세요.",
                        type=["pdf", "txt", "docx"],
                    )
                    if file:
                        docs = split_file(file)
                except:
                    st.error("파일 업로드에 실패하였습니다.")
            else:
                topic = st.text_input("위키피디아에서 검색할 주제를 입력하세요.")
                if topic:
                    docs = wiki_search(topic)

        else:
            st.warning("올바른 OPENAI_API_KEY를 입력하세요.")
            key = ""

if not docs:
    st.markdown(
        """
        AI가 내는 퀴즈를 맞춰보세요!\n
        
        *퀴즈는 업로드하시는 파일 혹은 위키피디아의 정보를 통해 만들어집니다.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, total_count, difficulty)
    response = response.additional_kwargs["function_call"]["arguments"]
    # with st.container():
    with st.form("questions_form"):
        for question in json.loads(response)["questions"]:
            current = current + 1
            # st.write(f'{total_count}. {question["question"]}')
            value = st.radio(
                f'Q{current}. {question["question"]}',
                [answer["answer"] for answer in question["answers"]],
                index=None, horizontal=True
            )
            st.markdown(
                """<style>
                    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
                        font-size: 18px;
                    }
                </style>
                """, unsafe_allow_html=True
            )
            
            if {"answer": value, "correct": True} in question["answers"]:
                st.success(f"Correct! ({value})")
                count = count + 1
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button(label="제출", type="primary", use_container_width=True, disabled=(True if count == total_count else False))

        if button:
            if count == total_count:
                st.info("만점이에요. 축하합니다!", icon="✅")
                st.balloons()
            else:
                st.warning(f"정답개수 : {count} / {total_count}. 다시 풀어보세요!", icon="❌")