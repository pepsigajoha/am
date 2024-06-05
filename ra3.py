import os
import streamlit as st
import joblib
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import google.generativeai as genai
import datetime
import time

# 환경 변수 로드 및 설정
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyBXG2DNlMikTUkFI5m8gylVLeTnruudl8g"
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

# PDF 문서에서 텍스트 추출
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 텍스트 청크 생성
def get_text_chunks(text):
    chunks = text_splitter.split_text(text)
    return chunks

# 벡터 스토어 생성 및 저장
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# 대화 체인 생성
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say anything. if so long, only print 1 line
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# 채팅 이력 초기화
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

# 사용자 입력 처리 및 응답 생성
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

# 문서 로드 및 처리
pdf_path = "C:/Users/jungj/OneDrive/바탕 화면/M1-Harry/전자금융_감독규정_해설.pdf"
def load_document_text(pdf_path: str) -> list[Document]:
    pdf_loader = PyPDFLoader(pdf_path)
    return pdf_loader.load_and_split()

# 메시지 파싱
def parse_message(string):
    return str(string).replace("\\n", "<br>")

# 이전 챗 로드 (있으면 로드, 없으면 초기화)
try:
    past_chats = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}
    
# 메인 함수
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    # 사이드바 설정
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


    # 세션 상태 관리
    if 'chat_id' not in st.session_state:
        st.session_state['chat_id'] = "New_Chat"

    chat_id_options = ["New_Chat"] + list(past_chats.keys())
    chat_id = st.selectbox('Pick a past chat', options=chat_id_options, index=0)

    if chat_id != "New_Chat":
        st.session_state.chat_id = chat_id
    else:
        st.session_state.chat_id = f"ChatSession-{int(time.time())}"

    st.session_state.chat_title = st.session_state.chat_id

    # 메인 채팅 인터페이스
    st.title('Financial Chatbot')
    st.write('Welcome to your financial assistant chatbot.')


    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                responses = user_input(prompt)
                placeholder = st.empty()
                full_response = ' '.join(responses)
                placeholder.markdown(full_response)
        if full_response:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

    # 대화 이력 저장
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-messages')
    joblib.dump(past_chats, 'data/past_chats_list')

if __name__ == "__main__":
    main()
