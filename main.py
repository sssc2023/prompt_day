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

# 제목
st.title("SightnSpeak")
st.title("가보자고")
st.write("---")

# 방 이미지
cyworld_img = Image.open('picture/livingroom.jpg')
# 이미지 크기 조정
cyworld_img = cyworld_img.resize((650, int(650 * (cyworld_img.height / cyworld_img.width))))
st.image(cyworld_img, width=650)
st.write("---")

db_ac = Chroma(persist_directory='./ac', embedding_function=OpenAIEmbeddings())
db_tv = Chroma(persist_directory='./tv', embedding_function=OpenAIEmbeddings())
db_hm = Chroma(persist_directory='./hm', embedding_function=OpenAIEmbeddings())
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def wrap_text(text, line_length=18):  # 챗봇 글자수 조절..
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
        st.success("가습기가 선택되었습니다.")
        st.session_state.selected_device = 'HM'

st.write("---")

# 질문하기 창이 나타나는 조건을 추가
# Air Conditioner
if st.session_state.selected_device == 'AC':
    st.subheader("❄️에어컨에게 질문해보세요!")
    ac_img = Image.open('picture/air-conditioner.png')
    ac_img = ac_img.resize((100, 100))
    st.image(ac_img)
    ac_question = st.text_input('안녕하세요, 전 에어컨이에요. 슝슝~', key='ac')
    st.write("---")
    with st.spinner('Wait for it...'):
        qa_chain_ac = RetrievalQA.from_chain_type(llm, retriever=db_ac.as_retriever())
        if ac_question != "":
            result = qa_chain_ac({"query": ac_question + '대답을 다 마치고 슝슝!이라고 말해줘'})
            st.session_state.chat_history['AC'].append({"question": ac_question, "answer": result["result"]})

    # 챗 기록 출력
    for chat in st.session_state.chat_history['AC']:
        st.text(f"🤔 {chat['question']}")
        st.text(f"😊 {chat['answer']}")
        st.write("---")

# TV
elif st.session_state.selected_device == 'TV':
    st.subheader("📺TV에게 질문해보세요!")
    tv_img = Image.open('picture/television.png')
    tv_img = tv_img.resize((100, 100))
    st.image(tv_img)
    tv_question = st.text_input('텔레비전에게 물어봐티비~')
    st.write("---")
    with st.spinner('Wait for it...'):
        qa_chain_tv = RetrievalQA.from_chain_type(llm, retriever=db_tv.as_retriever())
        if tv_question != "":
            result = qa_chain_tv({"query": tv_question + '대답을 다 마치고 떼레비!라고 말해줘'})
            st.session_state.chat_history['TV'].append({"question": tv_question, "answer": result["result"]})

    # 챗 기록 출력
    for chat in st.session_state.chat_history['TV']:
        st.text(f"🤔 {chat['question']}")
        st.text(f"😊 {chat['answer']}")
        st.write("---")

# Humidifier
elif st.session_state.selected_device == 'HM':
    st.subheader("💧가습기에게 질문해보세요!")
    hm_img = Image.open('picture/humidifier.png')
    hm_img = hm_img.resize((100, 100))
    st.image(hm_img)
    hm_question = st.text_input('안녕? 내가 아는 모든 걸 촉촉하게 알려줄게!', key='hm')
    st.write("---")
    with st.spinner('Wait for it...'):
        qa_chain_hm = RetrievalQA.from_chain_type(llm, retriever=db_hm.as_retriever())
        if hm_question != "":
            result = qa_chain_hm({"query": hm_question + '대답을 다 마치고 축축!이라고 말해줘'})
            st.session_state.chat_history['HM'].append({"question": hm_question, "answer": result["result"]})

    # 챗 기록 출력
    for chat in st.session_state.chat_history['HM']:
        st.text(f"🤔 {chat['question']}")
        st.text(f"😊 {chat['answer']}")
        st.write("---")
