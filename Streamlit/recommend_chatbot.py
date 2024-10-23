import streamlit as st 
import pandas as pd
import torch
import os
import json
from PIL import Image
##################################################################api
from dotenv import load_dotenv
import google.generativeai as genai
##################################################################vectordb
from langchain_chroma import Chroma
##################################################################embedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import(
    RunnableLambda
)
from langchain_core.messages import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from defs import (clear_chat_history, main, generate_random_id, reset_session_state, making_id,
                  get_session_history, category_classification, print_messages)

#.env 파일 생성해서 GEMINI_API_KEY=API_KEY 입력 후 실행하시면돼요
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

##################################################################
#chatbot UI
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구와 제주도에 오셨나요? 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

########################메시지 초기화 구간#####################

# 초기 메시지 시작할 때에 message container 만들어 이곳에 앞으로 저장
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] # 아예 내용을 지우고 싶다면 리스트 안의 내용을 clear 해주면 된다.

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if 'store' not in st.session_state:
    st.session_state['store'] = dict()

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = None

if "message_displayed" not in st.session_state:
    st.session_state.message_displayed = False
        
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('닉네임 생성'):
            making_id()
        
    with col2:
        if st.button('로그아웃'):
            reset_session_state()
            st.rerun()
            
    session_id = st.text_input('Session ID', value=st.session_state['session_id'], key='session_id')
     
    if st.button('로그인'):
            if st.session_state.session_id:
                st.session_state['is_logged_in'] = True  # 로그인 상태 저장
                get_session_history(session_id)
            else:
                st.sidebar.write("Session ID를 입력하세요.")
            
            # 로그인 성공 후 세션 상태 확인
            if st.session_state.get('is_logged_in', False):
                st.sidebar.write(f"현재 로그인된 닉네임: {st.session_state['session_id']}")
            else:
                st.sidebar.write("로그인 필요")
                   
    st.title('<옵션을 선택하면 빠르게 추천해드려요!>')
    st.write("")
    
    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴께요.')
    st.write("")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")
    
    # PNG 이미지 삽입 (제주도 지도.png 이미지 삽입!!!!!!!!!!!!)
    image = Image.open(r'D:\2024_bigcontest\data\이미지\제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입



# Display the initial assistant message only once
if st.session_state.is_logged_in and not st.session_state.message_displayed:
    st.session_state.messages.append({"role": "assistant", "content": "어떤 식당을 찾으시나요?"})
    st.session_state.message_displayed = True
    
# for message in st.session_state['messages']:  
#     with st.chat_message(message["role"]):  
#         st.write(message["content"])  
               
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
########################### 검색형 데이터 csv ########################### 
# 해당 데이터 경로로 변경 하세요!!   
path = r'D:\2024_bigcontest\data\final_data\JEJU_MCT_DATA_v2(12월)_v2.csv'
raw = pd.read_csv(path, index_col = 0)
df = raw.copy()
#########################임베딩 모델 로드##############################    
embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
#############################ChromaDB##############################    
# ChromaDB 불러오기
# 해당 데이터 경로로 변경 하세요!!
recommendation_store = Chroma(
    collection_name='jeju_store_mct_keyword_6',
    embedding_function=embedding_function,
    persist_directory= r'D:\2024_bigcontest\VectorDB\mct_keyword_v6'
)
# metadata 설정
metadata = recommendation_store.get(include=['metadatas'])

############################# 이전 대화기록 누적/출력해 주는 코드 #############################
# 이전 대화기로기을 출력해 주는 코드
print_messages()
get_session_history(session_id) 
###########################################사용자 입력 쿼리################################################
# 사용자 입력에 따른 검색
if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
        # ID가 없을 때 안내 메시지
        st.chat_message('assistant').write("닉네임을 생성하고 로그인해주세요.")
else:
    if user_input := st.chat_input():
        # 사용자가 입력한 내용을 출력
        st.chat_message('user').write(f'{user_input}') 
        st.session_state.messages.append({"role": "user", "content": user_input})
        
            
        with st.spinner("음식점을 찾는 중입니다..."):   
            
            # 음식점 검색 및 결과 반환
            response = main(user_input, df, session_id)
            
            # 사용자가 제주시를 선택하고 서귀포시 음식점을 요청한 경우
            if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in user_input):
                assistant_response = "제주시에 있는 음식점만 추천해드릴 수 있어요. 서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요."
            
            # 사용자가 서귀포시를 선택하고 제주시에 있는 음식점을 요청한 경우
            elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in user_input):
                assistant_response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요. 제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."

            # 검색 결과가 있는 경우
            elif response:
                assistant_response = response  # 검색 결과를 assistant_response로 저장
            
            # 검색 결과가 없을 때
            else:
                assistant_response = "질문해주신 음식점을 찾지 못했습니다. 다시 질문해주세요."

        # 챗봇 응답 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # 챗봇 응답 출력
        with st.chat_message("assistant"):
            # 세션 기록을 기반으로 한 RunnableWithMessageHistory 설정
                        process_user_query_runnable = RunnableLambda(
                            lambda inputs: main(inputs["question"], df, inputs["session_id"])
                        )

                        # 실제 RunnableWithMessageHistory 가 적용된 Chain
                        with_message_history = RunnableWithMessageHistory(
                            process_user_query_runnable,
                            get_session_history,
                            input_messages_key="question",
                            history_messages_key="history",
                        )
                        
                        # 질문이 들어오면 실행 (chain 실행)
                        response = with_message_history.invoke(
                            {"question": user_input, "session_id": st.session_state['session_id']},  
                            config={"configurable": {"session_id": st.session_state['session_id']}}
                        )
                    
                        # 최종 invoke한 내용을 response에 넣었고 그것을 contents에 저장
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        