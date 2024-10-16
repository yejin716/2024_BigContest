import streamlit as st 
import pandas as pd
import torch
import os
from PIL import Image
##################################################################api
from dotenv import load_dotenv
import google.generativeai as genai
##################################################################vectordb
import chromadb
from langchain_chroma import Chroma
##################################################################embedding
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

##################################################################
#chatbot UI
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구랑 제주도 왔나요? 맞춤 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

with st.sidebar:
    st.title('<옵션을 선택하면 빠르게 추천해드려요!>')
    st.write("")
    
    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴께요.')
    st.write("")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")
    
    st.subheader("맛집을 골라보세요! 관광객을 위한 맛집 또는 현지인들이 사랑하는 맛집을 선택할 수 있어요.")
    st.write("")
    
    # 체크박스 사용
    local_favorites = st.checkbox('제주도민 맛집')  # 제주도민 맛집 체크박스
    tourist_favorites = st.checkbox('관광객 맛집')  # 관광객 맛집 체크박스
    
    st.write('')
    # PNG 이미지 삽입
    image = Image.open(r'D:\Yebang\study\2024빅콘테스트\생성형AI분야\Code\Streamlit\제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
  
##################################################################    
#llm 함수 
def get_llm():
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    llm = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    return llm

#########################임베딩 모델 로드##############################    
@st.cache_resource
def load_embedding_model():
    model_name = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    return tokenizer, model, embedding_function

tokenizer, model, embedding_function = load_embedding_model()

#########################임베딩 함수##############################    
# 텍스트 임베딩
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

###########################ChromaDB#############################
search_store = Chroma(
    collection_name="jeju_store_mct_keyword_v4",
    embedding_function=embedding_function,
    persist_directory= r'D:\Yebang\study\2024빅콘테스트\생성형AI분야\VectorDB\mct_keyword_v3'
)  
###########################retreiver##########################
def create_filter(local_jeju_city, local_seogwipo_city, local_favorites, tourist_favorites):
    filters = {}

    # 2가지가 동시에 체크된 경우 처리
    if local_jeju_city and local_favorites: #제주시 + 현지인 맛집
        filters["address"] = {"$contains": "제주시"}
        filters["현지인"] = {"$contains": 1}  
    elif local_jeju_city and tourist_favorites: #제주시 + 관광객 맛집
        filters["address"] = {"$contains": "제주시"}
        filters["현지인"] = {"$contains": 0}  
    elif local_seogwipo_city and local_favorites: #서귀포시 + 현지인 맛집
        filters["address"] = {"$contains": "서귀포시"}
        filters["현지인"] = {"$contains": 1}  
    elif local_seogwipo_city and tourist_favorites:
        filters["address"] = {"$contains": "서귀포시"} #서귀포시 + 관광객 맛집
        filters["현지인"] = {"$contains": 0}  

    # 1개만 체크된 경우 처리
    elif local_jeju_city:
        filters["address"] = {"$contains": "제주시"}  # 제주시
    elif local_seogwipo_city:
        filters["address"] = {"$contains": "서귀포시"}  # 서귀포시
    elif local_favorites:
        filters["현지인"] = {"$contains": 1}  # 현지인 맛집
    elif tourist_favorites:
        filters["현지인"] = {"$contains": 0}  # 관광객 맛집
    return filters

# 필터링 조건에 따라 retriever 생성 함수
def get_retriever(local_jeju_city, local_seogwipo_city, local_favorites, tourist_favorites):
    filters = create_filter(local_jeju_city, local_seogwipo_city, local_favorites, tourist_favorites)

    search_retriever = search_store.as_retriever(
        search_kwargs={
            'k': 3,
            'filter': filters  # 필터 조건 적용
        }
    )
    return search_retriever

# 사용자가 선택한 체크박스 상태로 retriever 가져오기
search_retriever = get_retriever(local_jeju_city, local_seogwipo_city, local_favorites, tourist_favorites)
##################################################################
#  User-provided prompt
# 사용자 입력에 따른 검색
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 사용자 입력을 임베딩하여 ChromaDB에서 유사 문서 검색
    user_input_embedding = embed_text(user_input)  # 사용자 입력 임베딩
    results = search_store.similarity_search_by_vector(user_input_embedding, k=3)  # 유사 문서 검색

    # 검색 결과 처리 및 출력
    if results:
        with st.chat_message("assistant"):
            for result in results:
                st.write(f"추천 맛집: {result.metadata['name']}")
                st.write(f"주소: {result.metadata['address']}")
                st.write(f"요약: {result.metadata['summary']}")
    else:
        with st.chat_message("assistant"):
            st.write("죄송합니다. 선택하신 조건에 맞는 맛집을 찾을 수 없습니다.")