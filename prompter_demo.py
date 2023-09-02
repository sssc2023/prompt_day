#!/usr/bin/env python3

import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import numpy as np
import time
import math
import os
from _private import username, password, broker_address, sssc_gpt_api
from _chatGPT_STT import get_device_name, text_to_speech, text_to_speech_en, text_to_speech_set
from _save_txt_file import write_angleMap, read_angleMap

import openai
import speech_recognition as sr
import playsound
from gtts import gTTS

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = sssc_gpt_api

################### MQTT ####################
def on_connect(client, userdata, flags, rc):
    # rc: result code
    print("MQTT - Connected with result code " + str(rc))
    if rc == 0:
        print("MQTT - connected OK")
        mqttc.subscribe("register")
        mqttc.subscribe("select")
        mqttc.subscribe("control")
        mqttc.subscribe("chat")

    else:
        print("MQTT - Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print("MQTT - Disconnected with result code " + str(rc))

def on_publish(client, userdata, mid):
    print("MQTT - publish topic and message, pub#: ", mid)

def on_subscribe(client, obj, mid, granted_qos):
    print("MQTT - Subscribe complete: " + str(mid) + ", qos: " + str(granted_qos))

def start_mqtt():
    global mqttc
    mqttc = mqtt.Client("python client")

    # 연결 및 콜백 함수 설정
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.on_disconnect = on_disconnect
    mqttc.on_publish = on_publish
    mqttc.on_subscribe = on_subscribe

    mqttc.username_pw_set(username, password)
    mqttc.connect(broker_address, 1883)

    return mqttc

#################################################

def on_message(client, userdata, msg):
    # 메시지를 받았을 때 실행할 코드 작성
    print("MQTT - received topic: ", msg.topic)
    if msg.topic == "register":
        global device, start_time
        try:
            new_device = get_device_name()
        except:
            print("OpenAI - 기기 이름이 잘 인식되지 않음")
            playsound.playsound(f'fail.mp3')
        else:
            device = new_device
            start_time = time.time()  # start_time 업데이트
            playsound.playsound(f'looking_{device}.mp3')  # 기기 쳐다 봐야 하는 신호

            try:  # 에러 발생해도 안멈추게
                initialize()
            except:
                print("Mediapipe - 얼굴 인식 등의 문제로 기기 등록 실패")
            else:  # 에러 발생 안할 때만
                print(device)
                text_to_speech(device)
                playsound.playsound(f'setting_{device}.mp3')
                client.publish("state", "complete")  # 저장 완료

    elif msg.topic == "select":
        global pred_device
        pred_device = mapping()
        print("시선 예측 결과 기기 : ", pred_device)
        playsound.playsound(f'select_{pred_device}.mp3')

    elif msg.topic == "chat":
        chatbot(pred_device)

############## device mapping ###############
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def initialize():
    global buffer

    average_head_direction = np.mean(buffer, axis=0)
    write_angleMap(device, average_head_direction)
    print("device initializing end")

def mapping():
    device_map, angle_map = read_angleMap()
    angle_map = np.array(angle_map, dtype=np.float64)

    global p
    dis_arr = [euclaideanDistance(point, p) for point in angle_map]
    pred_device = device_map[np.argmin(dis_arr)]
    print("pred_device: ", pred_device)

    return pred_device

################################################
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
    embeddings_model = OpenAIEmbeddings(openai_api_key=sssc_gpt_api)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)
    return db

def chatbot(pred_device):
    # OpenAI API 설정
    openai.api_key = sssc_gpt_api
    # 음성 인식 준비
    recognizer = sr.Recognizer()

    # 마이크를 사용하여 음성 입력 받음
    with sr.Microphone(device_index=1) as source:
        playsound.playsound('tell_me.mp3')
        print("Please speak something:")
        audio = recognizer.listen(source, timeout=5)

    try:
        # Google Web Speech API를 사용하여 음성을 텍스트로 변환
        text = recognizer.recognize_google(audio, language='ko-KR')
        print(f"You said: {text}")

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Sorry, I could not request results; please check your network connection.")
    except sr.WaitTimeoutError:
        print("Sorry, You have to answer in time")

    else:
        # OpenAI ChatGPT 모델 사용해 keyword extraction 진행
        system_message = "너는 스마트홈 허브야. 메세지을 두 가지 종류로 분류해. 만약 {tv켜줘, 에어컨 온도 줄여줘}처럼 사용자가 기기를 조작하고 싶어하면 {control: 답변}으로 대답하고, 만약{tv 켜는 법, 에어컨 온도 조절하는 법}처럼 사용자가 기기의 매뉴얼을 궁금해 하면 {pdf : 답변}, 그 외의 메세지면{else: 답변}으로 대답하고 계속 다른 질문은 없냐고 물어봐줘. "
        control_message = "메세지에서 사용자가 어떤 조작을 원하는지 알려줘. 예를 들어, {tv 볼륨을 올려줘}라고 하면 {volume up}이라고 대답해주고, {온도 좀 높여줘}라고 하면 {temperature up}, {꺼줘}라고 하면 {power off}처럼 키워드로 대답해줘"
        user_message = text

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        # 답변
        answer = completion.choices[0].message.content
        print(answer)

        if "else" in answer: #계속 질문하게 하기?
            tts0 = gTTS(text=answer, lang='ko')
            tts0.save('chat_else.mp3')
            playsound.playsound('chat_else.mp3')
            os.remove('chat_else.mp3')

        elif "control" in answer:
            completion2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": control_message},
                    {"role": "user", "content": user_message}
                ]
            )
            control = completion2.choices[0].message.content
            print(control)

            tts0 = gTTS(text=control, lang='ko')
            tts0.save('chat_control.mp3')
            playsound.playsound('chat_control.mp3')
            os.remove('chat_control.mp3')

        elif "pdf" in answer: # pdf에서 답 찾기
            if pred_device == 'television':
                result = qa_chain_tv({"query": user_message})
            elif pred_device == 'airconditioner':
                result = qa_chain_ac({"query": user_message})
            elif pred_device == 'humidifier':
                result = qa_chain_hm({"query": user_message})

            print(result)
            tts0 = gTTS(text=result['result'], lang='ko')
            tts0.save('chat_pdf.mp3')
            playsound.playsound('chat_pdf.mp3')
            os.remove('chat_pdf.mp3')

    return 0

db_ac = Chroma(persist_directory='./ac', embedding_function=OpenAIEmbeddings())
db_tv = Chroma(persist_directory='./tv', embedding_function=OpenAIEmbeddings())
db_hm = Chroma(persist_directory='./hm', embedding_function=OpenAIEmbeddings())
llm = ChatOpenAI(openai_api_key=sssc_gpt_api, model_name="gpt-3.5-turbo", temperature=0)

qa_chain_tv = RetrievalQA.from_chain_type(llm, retriever=db_tv.as_retriever())
qa_chain_ac = RetrievalQA.from_chain_type(llm, retriever=db_ac.as_retriever())
qa_chain_hm = RetrievalQA.from_chain_type(llm, retriever=db_hm.as_retriever())



######## 시작 전에 필요한거 불러오기 #######
FACE = [1, 33, 61, 199, 263, 291]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                  refine_landmarks=True, max_num_faces=2)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

device = None
start_time = None
pred_device = None
text_to_speech_set(None)

head_directions = {}
buffer = []

client = start_mqtt()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # MQTT client 계속 실행
    client.loop_start()

    success, image = cap.read()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # To improve performance
    image.flags.writeable = False
    # Get the result
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    # To improve performance
    image.flags.writeable = True

    img_h, img_w, img_c = image.shape
    order = 0

    # Camera internals
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]], dtype="double"
                          )
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    if results.multi_face_landmarks:
        for single_face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=single_face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

            for idx, lm in enumerate(single_face_landmarks.landmark):
                if idx in FACE:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

        # Solve PnP face
        (_, rot_vec, trans_vec) = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        angles = np.array(angles) * 360
        # print("angles:\n {0}".format(angles))
        x = angles[0]
        y = angles[1]
        z = angles[2]

        p = [x, y]
        buffer.append((x, y))

        if len(buffer) > 10:
            buffer = buffer[-10:]

        # Add the text on the image
        cv2.putText(image, "x: " + str(np.round(x, 2)), (400, 50 + 200 * order), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 2)), (400, 100 + 200 * order), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (400, 150 + 200 * order), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        if device != None and start_time is not None and time.time() - start_time < 3:
            cv2.putText(image, f'device name: {device}', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if pred_device != None:
            cv2.putText(image, f'You select {pred_device}', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 누를 시 종료, == ord('문자')로 교체가능
        break

cap.release()