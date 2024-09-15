from typing_extensions import override
from openai import AssistantEventHandler
from openai import OpenAI
import streamlit as st
import requests, json
from utils import functions


############## streaming 처리를 위한 클래스
# 참고 : https://platform.openai.com/docs/assistants/tools/function-calling/step-3-initiate-a-run
class EventHandler(AssistantEventHandler):

    # 질문이 부실하거나 명확하지 않은 경우, 일반적인 답변을 streaming하기 위한 함수 제공
    # 예시 질의: fasfsafasfsakfasklfhaskhfdsakj
    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()
        
    def on_text_delta(self, delta, snapshot):
        self.message_box.markdown(snapshot.value)

    def on_text_done(self, text):
        save_message(text.value, "ai")
    
    #run의 status가 requires_action 일때 처리하는 로직 정의
    def on_event(self, event):
        # print(event.event)
        if event.event == "thread.run.requires_action":
            # submit 등 이후 처리를 위해 run.id를 event에서 가져옴
            run_id = event.data.id
            self.submit_tool_outputs(run_id, thread.id)

    # run의 정보를 가져옴
    def get_run(self, run_id, thread_id):
        return client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )
    
    # required_action에서 요구되는 함수가 실행될 수 있도록 매핑 수행
    def get_tool_outputs(self, run_id, thread_id):
        run = self.get_run(run_id, thread.id)
        outputs = []

        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            outputs.append(
                {
                    "output": functions.functions_map[function.name](json.loads(function.arguments)),
                    "tool_call_id": action_id,
                }
            )
        return outputs
    
    # get_tool_outputs()를 통해 가져온 정보를 streaming 처리
    def submit_tool_outputs(self, run_id, thread_id):
        outputs = self.get_tool_outputs(run_id, thread_id)
        with client.beta.threads.runs.submit_tool_outputs_stream(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=outputs,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
 

############## OPENAI_API_KEY 정합성 체크
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
    

############## assistant 생성
def init_assistant():
    ASSISTANT_NAME = "ggomdong's Research Assistant v1.0"

    # 동일한 assistant가 여러개 생성되는 것을 방지하기 위해 기존 assistant 정보를 가져옴
    # 동일한 assistant가 존재할 경우 해당 assistant를 반환
    my_assistants = client.beta.assistants.list(order="desc", limit="20")
    for assistant in my_assistants:
        if assistant.name == ASSISTANT_NAME:
            return client.beta.assistants.retrieve(assistant.id)
    
    # 없으면 신규로 생성
    assistant = client.beta.assistants.create(
        name="ggomdong's Research Assistant v1.0",
        instructions="""
        당신은 웹사이트 검색 및 조사 전문가입니다.

        사용자의 질의에 대해 Wikipedia 또는 DuckDuckGo 에서 완전하고 정확한 정보를 수집합니다.

        Wikipedia 또는 DuckDuckGo에서 적절한 웹사이트를 찾으면, 해당 웹사이트의 컨텐츠를 스크랩해야 합니다. 스크랩한 컨텐츠를 사용하여 질문에 대한 자세한 답변을 철저히 조사하고 형식화하세요.

        Wikipedia와 DuckDuckGo에서 검색하여 찾은 관련 웹사이트의 정보를 결합합니다. 최종 답변이 잘 정리되고 상세하며, 관련 링크(URL)와 함께 자료의 출처가 정확한지 여부와 잘 포함되어 있는지 여부를 확인합니다.

        관련 링크(URL)는 영어로 제공합니다.
        관련 링크(URL)를 제외한 최종 답변 및 파일 내용은 모두 한글이어야 합니다.

        링크와 출처는 가장 마지막에 표기합니다. 관련 링크 예시) Wikipedia: https://en.wikipedia.org/wiki/PlayStation_4

        최종 답변은 모든 출처와 관련 링크를 포함해 변경없이 동일하게 .txt 파일에 저장해야 합니다.
        """,
        tools=functions.functions,
        model="gpt-4o-mini-2024-07-18"
    )

    return assistant


############## 챗봇 메시지 처리를 위한 함수
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


############## streamlit 화면 영역
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍",
)

st.title("🔍 ResearchGPT")

st.markdown(
    """
    문의하신 내용을 Wikipedia 혹은 DuckDuckGo를 통해 조사해서 알려드립니다.
"""
)

with st.sidebar:
    st.markdown("""
    Github Repo : https://github.com/ggomdong/streamlit-gpt
    """)

    key = st.text_input("OPENAI_API_KEY", placeholder="OPENAI_API_KEY를 입력해주세요.", type="password")

    if key:
        # OPENAI_API_KEY 가 입력되면 파일 업로드 가능
        if is_valid(key):
            st.success("유효한 OPENAI_API_KEY 입니다.")
        else:
            st.warning("올바른 OPENAI_API_KEY를 입력하세요.")
            key = ""

if key:
    client = OpenAI(
        api_key=key
    )

    # assistant 초기화
    if "assistant" not in st.session_state:
        assistant = init_assistant()
        thread = client.beta.threads.create()
        
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    send_message("반갑습니다! 질문해 주세요. ^^", "ai", save=False)
    paint_history()
    message = st.chat_input("무엇이 궁금하신가요?")
    if message:
        send_message(message, "user")

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=message
        )
        try:
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
                ) as stream:
                with st.chat_message("ai"):
                    with st.spinner("처리중..."):
                        stream.until_done()
        except Exception as e:
            st.write(f"오류발생. {e}")
        
else:
    st.warning("OPENAI_API_KEY를 입력해주세요.")
    st.session_state["messages"] = []