import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
import pickle
import json
import pandas as pd
from datetime import datetime
import re
import glob
from pathlib import Path
from dotenv import load_dotenv

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Tư Vấn - Đại học Luật TPHCM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh nâng cao
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 20px;
        color: black;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .main-header h3 {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.8;
        position: relative;
        z-index: 1;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        color: #1565c0;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
        color: #6a1b9a;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .file-status {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .info-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 1px solid #ffb74d;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(255, 183, 77, 0.2);
    }
    
    .success-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 1px solid #4caf50;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(76, 175, 80, 0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
        border: 1px solid #ff9800;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(255, 152, 0, 0.2);
    }
    
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-tuyensinh {
        background: linear-gradient(135deg, #e3f2fd 0%, #90caf9 100%);
        color: #1565c0;
    }
    
    .badge-hocphi {
        background: linear-gradient(135deg, #e8f5e8 0%, #a5d6a7 100%);
        color: #2e7d32;
    }
    
    .badge-chuongtrinh {
        background: linear-gradient(135deg, #f3e5f5 0%, #ce93d8 100%);
        color: #6a1b9a;
    }
    
    .badge-sinhhoat {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
        color: #e65100;
    }
    
    .badge-hotro {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%);
        color: #c2185b;
    }
    
    .badge-totnghiep {
        background: linear-gradient(135deg, #e0f2f1 0%, #80cbc4 100%);
        color: #00695c;
    }
    
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .action-button {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        color: #495057;
    }
    
    .action-button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border-color: #2196f3;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .action-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-bottom: 1rem;
    }
    
    .footer-section h4 {
        color: #90caf9;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .footer-section p {
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
    }
    
    .chat-input {
        border-radius: 25px !important;
        border: 2px solid #e9ecef !important;
        padding: 1rem 1.5rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
    }
    
    .stChatInput > div > div > textarea {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e9ecef;
        border-top: 4px solid #2196f3;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #2196f3;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
#MainMenu {visibility: hidden;}
""", unsafe_allow_html=True)

# Load biến môi trường
load_dotenv()
grok_api_key = os.getenv("GROK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Cấu hình đường dẫn
DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"
CACHE_PATH = "cache"

# Tạo các thư mục nếu chưa tồn tại
for path in [DOCUMENTS_PATH, VECTORSTORE_PATH, CACHE_PATH]:
    Path(path).mkdir(exist_ok=True)

# Template prompt chuyên biệt cho tư vấn tuyển sinh
COUNSELING_PROMPT_TEMPLATE = """
Bạn là chuyên gia tư vấn tuyển sinh và công tác sinh viên của Trường Đại học Luật Thành phố Hồ Chí Minh.
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp và kiến thức chuyên môn.

Nguyên tắc trả lời:
1. Thân thiện, chuyên nghiệp và dễ hiểu
2. Cung cấp thông tin chính xác, cụ thể về Đại học Luật TPHCM
3. Đưa ra lời khuyên phù hợp với từng trường hợp
4. Hướng dẫn các bước cần thiết nếu có
5. Luôn khuyến khích và tạo động lực tích cực
6. Cung cấp thông tin liên hệ cụ thể khi cần thiết

Thông tin tham khảo: {context}

Lịch sử hội thoại: {chat_history}

Câu hỏi của sinh viên/thí sinh: {question}

Trả lời (bằng tiếng Việt, thân thiện và chuyên nghiệp):
"""

# Khởi tạo embeddings với model phù hợp tiếng Việt
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = load_embeddings()

# Hàm lấy danh sách file từ thư mục documents
def get_document_files():
    """Lấy danh sách tất cả file trong thư mục documents"""
    supported_extensions = ['*.pdf', '*.docx', '*.txt']
    files = []
    
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(DOCUMENTS_PATH, '**', ext), recursive=True))
    
    return files

# Hàm tạo hash cho file để kiểm tra thay đổi
def get_file_hash(file_path):
    """Tạo hash cho file để kiểm tra thay đổi"""
    stat = os.stat(file_path)
    return f"{stat.st_mtime}_{stat.st_size}"

# Hàm kiểm tra cache vector store
def load_cached_vectorstore():
    """Load vector store từ cache nếu có"""
    cache_file = os.path.join(VECTORSTORE_PATH, "vectorstore.pkl")
    metadata_file = os.path.join(VECTORSTORE_PATH, "metadata.json")
    
    if not (os.path.exists(cache_file) and os.path.exists(metadata_file)):
        return None, {}
    
    try:
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load vector store
        with open(cache_file, 'rb') as f:
            vectorstore = pickle.load(f)
        
        return vectorstore, metadata
    except Exception as e:
        st.warning(f"Không thể load cache: {e}")
        return None, {}

# Hàm lưu vector store vào cache
def save_vectorstore_cache(vectorstore, metadata):
    """Lưu vector store và metadata vào cache"""
    try:
        cache_file = os.path.join(VECTORSTORE_PATH, "vectorstore.pkl")
        metadata_file = os.path.join(VECTORSTORE_PATH, "metadata.json")
        
        # Lưu vector store
        with open(cache_file, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        # Lưu metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Lỗi lưu cache: {e}")
        return False

# Hàm kiểm tra xem có cần rebuild vector store không
def need_rebuild_vectorstore():
    """Kiểm tra xem có cần rebuild vector store không"""
    current_files = get_document_files()
    
    if not current_files:
        return False, {}, []
    
    # Tạo metadata hiện tại
    current_metadata = {}
    for file_path in current_files:
        current_metadata[file_path] = get_file_hash(file_path)
    
    # Load cached metadata
    _, cached_metadata = load_cached_vectorstore()
    
    # So sánh
    if current_metadata != cached_metadata.get('files', {}):
        return True, current_metadata, current_files
    
    return False, current_metadata, current_files

# Hàm xử lý file tài liệu
def process_documents(file_paths):
    """Xử lý danh sách file tài liệu"""
    documents = []
    processed_files = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                failed_files.append(f"{file_path} (không hỗ trợ)")
                continue
            
            docs = loader.load()
            
            # Thêm metadata
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(file_path)
                doc.metadata['file_path'] = file_path
                doc.metadata['processed_time'] = datetime.now().isoformat()
            
            documents.extend(docs)
            processed_files.append(file_path)
            
        except Exception as e:
            failed_files.append(f"{file_path} (lỗi: {str(e)})")
    
    return documents, processed_files, failed_files

# Hàm tạo vector store
def create_vector_store(documents):
    """Tạo vector store từ documents"""
    if not documents:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '.', '!', '?', ';', ':', ' ']
    )
    texts = text_splitter.split_documents(documents)
    
    # Lọc bỏ các chunk quá ngắn
    texts = [text for text in texts if len(text.page_content.strip()) > 50]
    
    if not texts:
        return None
        
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Hàm khởi tạo hoặc load vector store
@st.cache_resource
def initialize_vectorstore():
    """Khởi tạo hoặc load vector store"""
    need_rebuild, current_metadata, current_files = need_rebuild_vectorstore()
    
    if not need_rebuild:
        # Load từ cache
        vectorstore, cached_metadata = load_cached_vectorstore()
        if vectorstore:
            return vectorstore, cached_metadata.get('files', {}), cached_metadata.get('stats', {})
    
    # Rebuild vector store
    if not current_files:
        return None, {}, {}
    
    with st.spinner("🔄 Đang xử lý tài liệu..."):
        documents, processed_files, failed_files = process_documents(current_files)
        
        if not documents:
            return None, {}, {}
        
        vectorstore = create_vector_store(documents)
        
        if vectorstore:
            # Tạo metadata để lưu
            metadata_to_save = {
                'files': current_metadata,
                'stats': {
                    'total_files': len(current_files),
                    'processed_files': len(processed_files),
                    'failed_files': len(failed_files),
                    'total_chunks': vectorstore.index.ntotal,
                    'last_updated': datetime.now().isoformat()
                },
                'processed_files': processed_files,
                'failed_files': failed_files
            }
            
            # Lưu cache
            save_vectorstore_cache(vectorstore, metadata_to_save)
            
            return vectorstore, current_metadata, metadata_to_save['stats']
    
    return None, {}, {}

# Hàm phân loại câu hỏi
def classify_question(question):
    """Phân loại câu hỏi để đưa ra phản hồi phù hợp"""
    question_lower = question.lower()
    
    categories = {
        "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển", "kỳ thi", "thủ tục", "đăng kí", "nộp hồ sơ"],
        "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng", "trợ cấp", "tài chính", "phí", "tiền"],
        "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "khoa", "ngành", "thời khóa biểu", "học tập", "đào tạo"],
        "Sinh hoạt sinh viên": ["câu lạc bộ", "hoạt động", "thể thao", "văn hóa", "tình nguyện", "sinh hoạt", "sự kiện"],
        "Hỗ trợ sinh viên": ["tư vấn", "hỗ trợ", "ký túc xá", "thư viện", "cơ sở vật chất", "ktx", "ở", "chỗ ở"],
        "Tốt nghiệp": ["tốt nghiệp", "bằng cấp", "thực tập", "việc làm", "nghề nghiệp", "ra trường", "thực tế"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    return "Khác"

# Hàm tạo badge cho danh mục
def get_category_badge(category):
    """Tạo badge HTML cho danh mục câu hỏi"""
    badge_classes = {
        "Tuyển sinh": "badge-tuyensinh",
        "Học phí": "badge-hocphi", 
        "Chương trình đào tạo": "badge-chuongtrinh",
        "Sinh hoạt sinh viên": "badge-sinhhoat",
        "Hỗ trợ sinh viên": "badge-hotro",
        "Tốt nghiệp": "badge-totnghiep"
    }
    
    badge_class = badge_classes.get(category, "badge-tuyensinh")
    return f'<span class="category-badge {badge_class}">{category}</span>'

# Hàm khởi tạo Conversational Retrieval Chain
def create_conversational_chain(vector_store, llm):
    prompt = PromptTemplate(
        template=COUNSELING_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )
    
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Hàm kết nối LLM
@st.cache_resource
def get_gemini_llm():
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.3,
        max_output_tokens=1000
    )

@st.cache_resource
def get_grok_llm():
    return OpenAI(
        api_key=grok_api_key,
        base_url="https://api.x.ai/v1",
        model="grok-3",
        temperature=0.3,
        max_tokens=1000
    )

# Hàm trả lời từ API bên ngoài
def answer_from_external_api(prompt, llm, question_category):
    enhanced_prompt = f"""
    Bạn là chuyên gia tư vấn {question_category.lower()} của Trường Đại học Luật Thành phố Hồ Chí Minh.
    
    Câu hỏi: {prompt}
    
    Hãy trả lời một cách thân thiện, chuyên nghiệp và hữu ích. 
    Cung cấp thông tin chính xác về Đại học Luật TPHCM.
    Nếu không có thông tin cụ thể, hãy đưa ra lời khuyên chung phù hợp và 
    khuyến khích liên hệ phòng ban có liên quan để được hỗ trợ chi tiết hơn.
    
    Thông tin liên hệ:
    - Phòng Tuyển sinh: 1900 5555 14 hoặc 0879 5555 14
    - Phòng Công tác sinh viên:
    - Email: tuyensinh@hcmulaw.edu.vn
    - Địa chỉ: 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM
    """
    
    try:
        if isinstance(llm, GoogleGenerativeAI):
            response = llm.invoke(enhanced_prompt)
        else:
            response = llm.invoke(enhanced_prompt)
        return response
    except Exception as e:
        return f"Xin lỗi, tôi gặp một chút trục trặc kỹ thuật. Vui lòng thử lại sau hoặc liên hệ trực tiếp với phòng tư vấn theo số (028) 3838 5052. Lỗi: {str(e)}"

# Hàm lưu lịch sử hội thoại
def save_chat_history(user_question, bot_response, question_category):
    if 'chat_logs' not in st.session_state:
        st.session_state.chat_logs = []
    
    st.session_state.chat_logs.append({
        'timestamp': datetime.now().isoformat(),
        'user_question': user_question,
        'bot_response': bot_response,
        'category': question_category
    })

# Hàm hiển thị thống kê
def display_stats_cards(stats):
    """Hiển thị thống kê dưới dạng cards đẹp"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">📄 {stats.get('total_files', 0)}</div>
            <div class="metric-label">Tổng số file</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">✅ {stats.get('processed_files', 0)}</div>
            <div class="metric-label">Đã xử lý</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">📊 {stats.get('total_chunks', 0)}</div>
            <div class="metric-label">Chunks dữ liệu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">❌ {stats.get('failed_files', 0)}</div>
            <div class="metric-label">Lỗi xử lý</div>
        </div>
        """, unsafe_allow_html=True)

# Hàm hiển thị các câu hỏi gợi ý
def display_quick_questions():
    """Hiển thị các câu hỏi gợi ý"""
    st.markdown("### 💡 Câu hỏi thường gặp")
    
    quick_questions = [
        "📝 Thủ tục đăng ký xét tuyển như thế nào?",
        "💰 Học phí của trường là bao nhiều?", 
        "📚 Các ngành học của trường có gì?",
        "🏠 Trường có ký túc xá không?",
        "🎓 Cơ hội việc làm sau tốt nghiệp?",
        "📞 Thông tin liên hệ tư vấn?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.suggested_question = question.split(" ", 1)[1]  # Bỏ emoji

# Hàm hiển thị các tính năng
def display_features():
    """Hiển thị các tính năng của chatbot"""
    st.markdown("### 🚀 Tính năng hỗ trợ")
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Tư vấn tuyển sinh</div>
            <div class="feature-description">Hướng dẫn chi tiết về thủ tục đăng ký, điểm chuẩn, phương thức xét tuyển</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Hỗ trợ sinh viên</div>
            <div class="feature-description">Thông tin về ký túc xá, học bổng, hoạt động ngoại khóa</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Chương trình đào tạo</div>
            <div class="feature-description">Chi tiết về các ngành học, môn học, tín chỉ và kế hoạch học tập</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌟</div>
            <div class="feature-title">Tư vấn nghề nghiệp</div>
            <div class="feature-description">Định hướng nghề nghiệp, cơ hội việc làm sau tốt nghiệp</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Giao diện chính
def main():
    # Header với animation
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Chatbot Tư Vấn Tuyển Sinh</h1>
        <h3>Trường Đại học Luật Thành phố Hồ Chí Minh</h3>
        <p>🤖 Hỗ trợ 24/7 | 💬 Tư vấn chuyên nghiệp | 📞 Thông tin chính xác</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar cải tiến
    with st.sidebar:
        st.markdown("## 🛠️ Bảng điều khiển")
        
        # Thông tin hệ thống
        st.markdown("### 📊 Trạng thái hệ thống")
        
        # Khởi tạo vector store
        with st.spinner("🔄 Đang khởi tạo hệ thống..."):
            vectorstore, file_metadata, stats = initialize_vectorstore()
            st.session_state.vector_store = vectorstore
            st.session_state.file_stats = stats
        
        if stats:
            st.markdown("""
            <div class="success-card">
                <h4>✅ Hệ thống đã sẵn sàng!</h4>
                <p>Tất cả tài liệu đã được xử lý và sẵn sàng phục vụ.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị thống kê mini
            display_stats_cards(stats)
            
            # Thời gian cập nhật
            if 'last_updated' in stats:
                last_updated = datetime.fromisoformat(stats['last_updated'])
                st.markdown(f"""
                <div class="info-card">
                    <strong>🕒 Cập nhật cuối:</strong><br>
                    {last_updated.strftime('%d/%m/%Y lúc %H:%M')}
                </div>
                """, unsafe_allow_html=True)
            
            # Nút refresh với style
            if st.button("🔄 Làm mới tài liệu", type="primary", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        else:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Không tìm thấy tài liệu!</h4>
                <p>Vui lòng thêm file tài liệu vào thư mục <code>documents</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hướng dẫn chi tiết
            with st.expander("📋 Hướng dẫn thêm tài liệu", expanded=True):
                st.markdown(f"""
                **Các bước thêm tài liệu:**
                1. 📁 Tạo thư mục `{DOCUMENTS_PATH}` 
                2. 📄 Copy file PDF, DOCX, TXT vào thư mục
                3. 🔄 Nhấn "Làm mới tài liệu"
                
                **Định dạng hỗ trợ:**
                - 📄 PDF files (*.pdf)
                - 📝 Word documents (*.docx)  
                - 📋 Text files (*.txt)
                
                **Lưu ý:** File nên chứa thông tin về tuyển sinh, chương trình đào tạo, sinh hoạt sinh viên của trường.
                """)

        st.divider()
        
        # Lựa chọn AI model
        st.markdown("### 🤖 Cấu hình AI")
        llm_option = st.selectbox(
            "Chọn mô hình AI:", 
            ["Gemini", "Grok"],
            help="Gemini: Phù hợp cho câu hỏi chung\nGrok: Phù hợp cho phân tích chi tiết"
        )
        
        st.divider()
        
        # Thống kê chat
        st.markdown("### 📈 Thống kê phiên làm việc")
        if 'messages' in st.session_state and st.session_state.messages:
            total_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">💬 {total_messages}</div>
                <div class="metric-label">Câu hỏi đã hỏi</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Phân tích theo danh mục
            if hasattr(st.session_state, 'chat_logs') and st.session_state.chat_logs:
                categories = {}
                for log in st.session_state.chat_logs:
                    cat = log['category']
                    categories[cat] = categories.get(cat, 0) + 1
                
                st.markdown("**📊 Phân loại câu hỏi:**")
                for cat, count in categories.items():
                    st.markdown(f"• {cat}: {count} câu")
        else:
            st.markdown("""
            <div class="info-card">
                <p>Chưa có câu hỏi nào trong phiên này.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Thông tin liên hệ
        st.markdown("### 📞 Thông tin liên hệ")
        st.markdown("""
        <div class="info-card">
            <strong>🏛️ Đại học Luật TPHCM</strong><br>
            📍 2 Nguyễn Tất Thành, Q.4, TPHCM<br>
            📞 Tuyển sinh: 1900 5555 14 hoặc 0879 5555 14<br>
            📞 CTSV: (028) 39400 989<br>
            📧 tuyensinh@hcmulaw.edu.vn<br>
            🌐 www.hcmulaw.edu.vn
        </div>
        """, unsafe_allow_html=True)

    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Kiểm tra API keys
    if llm_option == "Gemini" and not gemini_api_key:
        st.error("⚠️ Vui lòng cung cấp GEMINI_API_KEY trong file .env")
        st.stop()
    elif llm_option == "Grok" and not grok_api_key:
        st.error("⚠️ Vui lòng cung cấp GROK_API_KEY trong file .env")
        st.stop()

    # Khởi tạo LLM
    if llm_option == "Gemini":
        llm = get_gemini_llm()
    else:
        llm = get_grok_llm()

    # Khởi tạo chain nếu có vector store
    chain = None
    if st.session_state.get('vector_store'):
        chain = create_conversational_chain(st.session_state.vector_store, llm)

    # Nội dung chính
    if not st.session_state.messages:
        # Trang chào mừng
        st.markdown("### 👋 Chào mừng bạn đến với Chatbot Tư Vấn!")
        
        # Hiển thị tính năng
        display_features()
        
        # Câu hỏi gợi ý
        display_quick_questions()
        
        # Hướng dẫn sử dụng
        st.markdown("""
        <div class="info-card">
            <h4>💡 Cách sử dụng hiệu quả:</h4>
            <ul>
                <li>🎯 Đặt câu hỏi cụ thể về lĩnh vực bạn quan tâm</li>
                <li>📝 Cung cấp thông tin chi tiết để được tư vấn chính xác</li>
                <li>🔄 Tiếp tục hỏi để làm rõ thêm thông tin</li>
                <li>📞 Liên hệ trực tiếp nếu cần hỗ trợ khẩn cấp</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị lịch sử chat với style mới
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "category" in message:
                # Hiển thị badge danh mục
                st.markdown(get_category_badge(message["category"]), unsafe_allow_html=True)
            st.markdown(message["content"])

    # Xử lý câu hỏi gợi ý
    if hasattr(st.session_state, 'suggested_question'):
        prompt = st.session_state.suggested_question
        del st.session_state.suggested_question
    else:
        prompt = st.chat_input("💬 Hãy đặt câu hỏi của bạn...")

    # Xử lý câu hỏi
    if prompt:
        # Hiển thị câu hỏi người dùng
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Phân loại câu hỏi
        question_category = classify_question(prompt)

        # Xử lý và trả lời
        with st.chat_message("assistant"):
            # Hiển thị badge danh mục
            st.markdown(get_category_badge(question_category), unsafe_allow_html=True)
            
            with st.spinner("🤔 Đang phân tích và tìm kiếm thông tin..."):
                try:
                    if chain and st.session_state.get('vector_store'):
                        # Sử dụng RAG với tài liệu
                        response = chain({"question": prompt})
                        answer = response["answer"]
                        
                        # Hiển thị nguồn tham khảo
                        if response.get("source_documents"):
                            st.markdown("---")
                            with st.expander("📚 Nguồn tham khảo từ tài liệu", expanded=False):
                                for i, doc in enumerate(response["source_documents"][:3]):
                                    st.markdown(f"""
                                    **📄 Nguồn {i+1}:** `{doc.metadata.get('source_file', 'N/A')}`
                                    
                                    *Nội dung:* {doc.page_content[:300]}...
                                    """)
                    else:
                        # Sử dụng AI thuần túy
                        answer = answer_from_external_api(prompt, llm, question_category)
                    
                    st.markdown(answer)
                    
                    # Lưu lịch sử
                    save_chat_history(prompt, answer, question_category)
                    
                except Exception as e:
                    error_msg = f"""
                    🔧 **Xin lỗi, hệ thống gặp sự cố kỹ thuật**
                    
                    Vui lòng thử lại sau hoặc liên hệ trực tiếp:
                    📞 **Hotline tư vấn:** (028) 3838 5052
                    📧 **Email:** tuyensinh@hcmulaw.edu.vn
                    
                    *Mã lỗi: {str(e)}*
                    """
                    st.error(error_msg)
                    answer = error_msg

        # Lưu tin nhắn với danh mục
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "category": question_category
        })

    # Footer chuyên nghiệp
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div class="footer-grid">
            <div class="footer-section">
                <h4>🏛️ Trường Đại học Luật TPHCM</h4>
                <p>📍 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM</p>
                <p>📞 Điện thoại: (028) 39400 989</p>
                <p>📧 Email: info@hcmulaw.edu.vn</p>
            </div>
            <div class="footer-section">
                <h4>📞 Hotline tư vấn</h4>
                <p>🎓 Tuyển sinh: (028) 39400 989</p>
                <p>👥 Công tác SV: ((028) 39400 989</p>
                <p>🏠 Ký túc xá: (028) 39400 989</p>
                <p>💰 Học phí: (028) 39400 989</p>
            </div>
            <div class="footer-section">
                <h4>🌐 Liên kết</h4>
                <p>🌍 Website: www.hcmulaw.edu.vn</p>
                <p>📘 Facebook: /hcmulaw</p>
                <p>📺 YouTube: /hcmulaw</p>
                <p>📧 Zalo: 09123456789</p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <p>🤖 <strong>Chatbot Tư Vấn</strong> - Phiên bản 2.0 | 🕒 Hỗ trợ 24/7 | 💬 Phản hồi tức thì</p>
            <p style="font-size: 0.8em; opacity: 0.8;">Được phát triển bởi Lvphung - CNTT - Đại học Luật TPHCM</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
