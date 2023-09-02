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

#파일 업로드
# ["samsung_tv_manual.pdf", "lg_ac_manual.pdf", "winix_humidifier_manual.pdf"]
tv_file = PyPDFLoader("samsung_tv_manual.pdf")
ac_file = PyPDFLoader("lg_ac_manual.pdf")
hm_file = PyPDFLoader("winix_humidifier_manual.pdf")

#제목
st.title("SightnSpeak")
st.write("---")

# 방 이미지
cyworld_img = Image.open('livingroom.jpg')
# 이미지 크기 조정
cyworld_img = cyworld_img.resize((650, int(650 * (cyworld_img.height / cyworld_img.width))))
st.image(cyworld_img, width=650)
st.write("---")

def document_to_db(uploaded_file, size):    # 문서 크기에 맞게 사이즈 지정하면 좋을 것 같아서 para 넣었어용
    pages = uploaded_file.load_and_split()
    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = size,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)
    return db

def wrap_text(text, line_length=18): # 챗봇 글자수 조절..
    lines = []
    for i in range(0, len(text), line_length):
        lines.append(text[i:i + line_length])
    return "\n".join(lines)


# 초기 세션 상태 설정
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {'AC': [], 'TV': [], 'HM': []}
if 'selected_device' not in st.session_state:
    st.session_state.selected_device = None

# 업로드 되면 동작하는 코드
if ac_file is not None and tv_file is not None and hm_file is not None:
    db_ac = document_to_db(ac_file, 500)
    db_tv = document_to_db(tv_file, 500)
    db_hm = document_to_db(hm_file, 300)

    # Choice
st.subheader("기기를 바라보고 선택하세요!")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("person_AC.jpg", width=100)
    st.markdown("에어컨을 <br/> 바라본다", unsafe_allow_html=True)
    if st.button("에어컨 선택"):
        st.write("에어컨이 선택되었습니다.")
        st.session_state.selected_device = 'AC'

with col2:
    st.image("person_TV.jpg", width=100)
    st.markdown("TV를 <br/> 바라본다", unsafe_allow_html=True)
    if st.button("TV 선택"):
        st.write("TV가 선택되었습니다.")
        st.session_state.selected_device = 'TV'

with col3:
    st.image("person_HM.jpg", width=100)
    st.markdown("가습기를 <br/> 바라본다", unsafe_allow_html=True)
    if st.button("가습기 선택"):
        st.write("가습기가 선택되었습니다.")
        st.session_state.selected_device = 'HM'

st.write("---")
col_ac, col_tv, col_hm = st.columns(3)
# 질문하기 창이 나타나는 조건을 추가
# Air Conditioner
if st.session_state.selected_device == 'AC':
    with col_ac:
        st.subheader("에어컨에게 질문해보세요!")
        ac_img = Image.open('air-conditioner.png')
        ac_img = ac_img.resize((100, 100))
        st.image(ac_img)
        ac_question = st.text_input('안녕하세요, 전 에어컨이에요. 슝슝~', key='ac')
        st.write("---")
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db_ac.as_retriever())
            result = qa_chain({"query": ac_question+'대답을 <슝슝>으로 끝내줘. 예시: 알겠슝슝'})
            st.session_state.chat_history['AC'].append({"question": ac_question, "answer": result["result"]})

        # 챗 기록 출력
        for chat in st.session_state.chat_history['AC']:
            st.text(f"🤔 {wrap_text(chat['question'])}")
            st.text(f"😊 {wrap_text(chat['answer'])}")
            st.write("---")

# TV
elif st.session_state.selected_device == 'TV':
    with col_tv:
        st.subheader("TV에게 질문해보세요!")
        tv_img = Image.open('television.png')
        tv_img = tv_img.resize((100, 100))
        st.image(tv_img)
        tv_question = st.text_input('텔레비전에게 물어봐티비~')
        st.write("---")
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db_tv.as_retriever())
            result = qa_chain({"query": tv_question+'대답을 <티비>로 끝내줘. 예시: 알겠티비' })
            st.session_state.chat_history['TV'].append({"question": tv_question, "answer": result["result"]})

        # 챗 기록 출력
        for chat in st.session_state.chat_history['TV']:
            st.text(f"🤔 {wrap_text(chat['question'])}")
            st.text(f"😊 {wrap_text(chat['answer'])}")
            st.write("---")

# Humidifier
elif st.session_state.selected_device == 'HM':
    with col_hm:
        st.subheader("가습기에게 질문해보세요!")
        hm_img = Image.open('humidifier.png')
        hm_img = hm_img.resize((100, 100))
        st.image(hm_img)
        hm_question = st.text_input('안녕? 내가 아는 모든걸  촉촉하게 알려줄게!', key='hm')
        st.write("---")
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db_hm.as_retriever())
            result = qa_chain({"query": hm_question+'대답을 <축축>으로 끝내줘. 예시: 알겠축축'})
            st.session_state.chat_history['HM'].append({"question": hm_question, "answer": result["result"]})

        # 챗 기록 출력
        for chat in st.session_state.chat_history['HM']:
            st.text(f"🤔 {wrap_text(chat['question'])}")
            st.text(f"😊 {wrap_text(chat['answer'])}")
            st.write("---")
