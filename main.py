__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from PIL import Image
import time
from langchain import PromptTemplate

# 제목
st.title("LookNTalk")
st.write("---")
st.write('이곳은 당신의 집 입니다.')
st.write('실제 서비스는 하드웨어(시선 추적용 카메라, 음성인식용 마이크 및 스피커)가 포함되어 있지만 이 MVP는 웹 상으로 시뮬레이션을 구현한것입니다. 하드웨어가 포함된 동작 영상을 참고해주세요. ')


# 방 이미지
room_img = Image.open('picture/living_room.png')
# 이미지 크기 조정
room_img = room_img.resize((650, int(650 * (room_img.height / room_img.width))))
st.image(room_img, width=650)
st.write("---")

db_ac = Chroma(persist_directory='./ac', embedding_function=OpenAIEmbeddings())
db_tv = Chroma(persist_directory='./tv', embedding_function=OpenAIEmbeddings())
db_hm = Chroma(persist_directory='./hm', embedding_function=OpenAIEmbeddings())
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def wrap_text(text, line_length=10):  # 챗봇 글자수 조절..
    lines = []
    for i in range(0, len(text), line_length):
        lines.append(text[i:i + line_length])
    return "\n".join(lines)
    
# 초기 세션 상태 설정
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {'AC': [], 'TV': [], 'HM': []}
if 'selected_device' not in st.session_state:
    st.session_state.selected_device = None

# Choice
st.subheader("선택할 기기를 바라보세요!")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("picture/person_AC.jpg", width=100)
    st.markdown("❄️에어컨을 바라본다", unsafe_allow_html=True)
    if st.button("에어컨 선택"):
        st.success("에어컨이 선택되었습니다.")
        st.session_state.selected_device = 'AC'

with col2:
    st.image("picture/person_TV.jpg", width=100)
    st.markdown("📺TV를 바라본다", unsafe_allow_html=True)
    if st.button("TV 선택"):
        st.success("TV가 선택되었습니다.")
        st.session_state.selected_device = 'TV'

with col3:
    st.image("picture/person_HM.jpg", width=100)
    st.markdown("💧가습기를 바라본다", unsafe_allow_html=True)
    if st.button("가습기 선택"):
        st.success("가습기가 선택되었습니다."촉~'을 붙여주세요.
        
        {context}

        질문: {question}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain_hm = RetrievalQA.from_chain_type(llm, retriever=db_hm.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        if hm_question != "":
            result = qa_chain_hm({"query": hm_question})
            st.session_state.chat_history['HM'].append({"question": hm_question, "answer": result["result"]})

    # 챗 기록 출력
    for chat in st.session_state.chat_history['HM']:
        st.text(f"🤔 {wrap_text(chat['question'])}")
        st.text(f"😊 {wrap_text(chat['answer'])}")
        st.write("---")
