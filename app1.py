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
from dotenv import load_dotenv
import tempfile
import json
import pandas as pd
from datetime import datetime
import re
import hashlib
import sqlite3
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import PyPDF2
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Tư Vấn - ĐH Luật TPHCM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh nâng cao
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
    }
    .sidebar-info {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .status-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .status-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f1aeb5 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load biến môi trường
load_dotenv()
grok_api_key = os.getenv("GROK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Database initialization
def init_database():
    """Khởi tạo database SQLite"""
    conn = sqlite3.connect('law_chatbot.db')
    cursor = conn.cursor()
    
    # Bảng người dùng
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Bảng lịch sử chat
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Bảng documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            uploaded_by INTEGER,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (uploaded_by) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Khởi tạo database
init_database()

# Hàm xác thực người dùng
def verify_user(username, password):
    """Xác thực người dùng"""
    conn = sqlite3.connect('law_chatbot.db')
    cursor = conn.cursor()
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT id, username, full_name, role FROM users WHERE username = ? AND password = ?', 
                   (username, hashed_password))
    user = cursor.fetchone()
    
    if user:
        # Cập nhật last_login
        cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user[0],))
        conn.commit()
    
    conn.close()
    return user

def register_user(username, password, full_name, email, role='student'):
    """Đăng ký người dùng mới"""
    conn = sqlite3.connect('law_chatbot.db')
    cursor = conn.cursor()
    
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password, full_name, email, role) 
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hashed_password, full_name, email, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_chat_to_db(user_id, question, answer, category):
    """Lưu lịch sử chat vào database"""
    conn = sqlite3.connect('law_chatbot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (user_id, question, answer, category) 
        VALUES (?, ?, ?, ?)
    ''', (user_id, question, answer, category))
    
    conn.commit()
    conn.close()

def get_user_chat_history(user_id, limit=50):
    """Lấy lịch sử chat của user"""
    conn = sqlite3.connect('law_chatbot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT question, answer, category, timestamp 
        FROM chat_history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (user_id, limit))
    
    history = cursor.fetchall()
    conn.close()
    return history

# Template prompt nâng cao
ENHANCED_COUNSELING_PROMPT = """
Bạn là AI Tư Vấn Viên chuyên nghiệp của Trường Đại học Luật Thành phố Hồ Chí Minh.

THÔNG TIN TRƯỜNG:
- Tên đầy đủ: Trường Đại học Luật Thành phố Hồ Chí Minh
- Địa chỉ: 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM
- Website: http://hcmulaw.edu.vn/
- Hotline: (028) 39 400 989

NGUYÊN TẮC TƢ VẤN:
1. 🎯 Thân thiện, chuyên nghiệp, tận tình
2. 📚 Cung cấp thông tin chính xác từ tài liệu có sẵn
3. 💡 Đưa ra lời khuyên cụ thể, phù hợp
4. 🔍 Hướng dẫn chi tiết các bước thực hiện
5. ❤️ Luôn động viên và tạo động lực tích cực
6. 📞 Hướng dẫn liên hệ trực tiếp khi cần thiết

THÔNG TIN THAM KHẢO: {context}

LỊCH SỬ HỘI THOẠI: {chat_history}

CÂU HỎI: {question}

HÃY TRẢ LỜI:
- Bằng tiếng Việt
- Cấu trúc rõ ràng với emoji phù hợp
- Cung cấp thông tin cụ thể và hữu ích
- Kết thúc với lời khuyên hoặc bước tiếp theo
"""

# Khởi tạo embeddings với xử lý lỗi
@st.cache_resource
def load_embeddings():
    """Load embeddings với xử lý lỗi và fallback"""
    try:
        # Thử model tiếng Việt tốt nhất
        return HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.warning(f"Không thể load vietnamese-sbert: {e}")
        try:
            # Fallback sang model khác
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e2:
            logger.error(f"Không thể load embeddings: {e2}")
            st.error("Lỗi khởi tạo embeddings. Vui lòng kiểm tra kết nối mạng.")
            return None

embeddings = load_embeddings()

# Hàm xử lý PDF cải tiến
def extract_text_from_pdf(file_path):
    """Trích xuất text từ PDF với nhiều phương pháp"""
    text = ""
    
    # Phương pháp 1: PyPDF2
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
    
    # Phương pháp 2: PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e}")
    
    return text

# Hàm phân loại câu hỏi nâng cao
def classify_question_advanced(question):
    """Phân loại câu hỏi với AI"""
    question_lower = question.lower()
    
    categories = {
        "Tuyển sinh": {
            "keywords": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển", "kỳ thi", "thủ tục", "đại học", "cao đẳng", "liên thông"],
            "priority": 1
        },
        "Học phí": {
            "keywords": ["học phí", "chi phí", "miễn giảm", "học bổng", "trợ cấp", "tài chính", "kinh phí", "thanh toán"],
            "priority": 2
        },
        "Chương trình đào tạo": {
            "keywords": ["chương trình", "môn học", "tín chỉ", "khoa", "ngành", "thời khóa biểu", "giảng viên", "giáo trình"],
            "priority": 3
        },
        "Sinh hoạt sinh viên": {
            "keywords": ["câu lạc bộ", "hoạt động", "thể thao", "văn hóa", "tình nguyện", "đoàn hội", "sự kiện"],
            "priority": 4
        },
        "Hỗ trợ sinh viên": {
            "keywords": ["tư vấn", "hỗ trợ", "ký túc xá", "thư viện", "cơ sở vật chất", "phòng ban", "dịch vụ"],
            "priority": 5
        },
        "Tốt nghiệp": {
            "keywords": ["tốt nghiệp", "bằng cấp", "thực tập", "việc làm", "nghề nghiệp", "luận văn", "khóa luận"],
            "priority": 6
        }
    }
    
    best_match = None
    max_matches = 0
    
    for category, info in categories.items():
        matches = sum(1 for keyword in info["keywords"] if keyword in question_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = category
    
    return best_match if best_match else "Tổng quát"

# Hàm xử lý file nâng cao
def process_uploaded_files_enhanced(uploaded_files):
    """Xử lý file với nhiều cải tiến"""
    documents = []
    processed_files = []
    error_files = []
    
    supported_extensions = {'.pdf': 'PDF', '.docx': 'Word', '.txt': 'Text'}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        file_extension = f".{file.name.split('.')[-1].lower()}"
        
        # Cập nhật progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Đang xử lý: {file.name}")
        
        if file_extension not in supported_extensions:
            error_files.append(f"{file.name} - Định dạng không hỗ trợ")
            continue
        
        try:
            # Tạo hash để kiểm tra trùng lặp
            file_content = file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Xử lý theo loại file
            if file_extension == ".pdf":
                text_content = extract_text_from_pdf(temp_file_path)
                if not text_content.strip():
                    error_files.append(f"{file.name} - Không thể trích xuất text từ PDF")
                    continue
                
                # Tạo document object
                from langchain.schema import Document
                doc = Document(
                    page_content=text_content,
                    metadata={
                        'source_file': file.name,
                        'file_type': 'PDF',
                        'file_hash': file_hash,
                        'upload_time': datetime.now().isoformat(),
                        'processed_by': 'enhanced_pdf_processor'
                    }
                )
                documents.append(doc)
                
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        'source_file': file.name,
                        'file_type': 'Word',
                        'file_hash': file_hash,
                        'upload_time': datetime.now().isoformat()
                    })
                documents.extend(docs)
                
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        'source_file': file.name,
                        'file_type': 'Text',
                        'file_hash': file_hash,
                        'upload_time': datetime.now().isoformat()
                    })
                documents.extend(docs)
            
            processed_files.append(file.name)
            os.unlink(temp_file_path)
            
        except Exception as e:
            error_files.append(f"{file.name} - Lỗi: {str(e)}")
            logger.error(f"Error processing {file.name}: {e}")

    progress_bar.empty()
    status_text.empty()
    
    return documents, processed_files, error_files

# Hàm tạo vector store cải tiến
def create_enhanced_vector_store(documents):
    """Tạo vector store với xử lý tối ưu"""
    if not documents or not embeddings:
        return None
    
    try:
        # Text splitter tối ưu cho tiếng Việt
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Giảm chunk size để tăng độ chính xác
            chunk_overlap=100,  # Tăng overlap
            separators=['\n\n', '\n', '.', '!', '?', ';', ':', ' ', ''],
            length_function=len,
        )
        
        # Split documents
        texts = text_splitter.split_documents(documents)
        
        # Lọc và làm sạch chunks
        clean_texts = []
        for text in texts:
            content = text.page_content.strip()
            # Loại bỏ chunks quá ngắn hoặc chỉ chứa ký tự đặc biệt
            if len(content) > 30 and not re.match(r'^[\s\W]*$', content):
                clean_texts.append(text)
        
        if not clean_texts:
            st.warning("Không có nội dung hợp lệ để tạo vector store")
            return None
        
        # Tạo vector store
        vector_store = FAISS.from_documents(clean_texts, embeddings)
        
        logger.info(f"Created vector store with {len(clean_texts)} chunks")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"Lỗi tạo vector store: {str(e)}")
        return None

# Hàm tạo conversational chain nâng cao
def create_enhanced_conversational_chain(vector_store, llm):
    """Tạo conversational chain với prompt tối ưu"""
    if not vector_store:
        return None
    
    prompt = PromptTemplate(
        template=ENHANCED_COUNSELING_PROMPT,
        input_variables=["context", "chat_history", "question"]
    )
    
    memory = ConversationBufferWindowMemory(
        k=10,  # Tăng memory window
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4, "fetch_k": 8}  # Tối ưu retrieval
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )

# LLM với xử lý lỗi
@st.cache_resource
def get_enhanced_gemini_llm():
    """Khởi tạo Gemini LLM với xử lý lỗi"""
    try:
        return GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.2,  # Giảm temperature để tăng tính nhất quán
            max_output_tokens=1500
        )
    except Exception as e:
        logger.error(f"Error initializing Gemini: {e}")
        return None

@st.cache_resource
def get_enhanced_grok_llm():
    """Khởi tạo Grok LLM với xử lý lỗi"""
    try:
        return OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1",
            model="grok-beta",
            temperature=0.2,
            max_tokens=1500
        )
    except Exception as e:
        logger.error(f"Error initializing Grok: {e}")
        return None

# Authentication UI
def show_login_page():
    """Hiển thị trang đăng nhập"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 Đăng Nhập Hệ Thống</h1>
        <p>Chatbot Tư Vấn - Đại học Luật TPHCM</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Đăng Nhập", "Đăng Ký"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Đăng Nhập")
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            submit = st.form_submit_button("Đăng Nhập", type="primary")
            
            if submit:
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.session_state.full_name = user[2]
                        st.session_state.role = user[3]
                        st.session_state.authenticated = True
                        st.success(f"Chào mừng {user[2]}!")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu không đúng!")
                else:
                    st.warning("Vui lòng nhập đầy đủ thông tin!")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Đăng Ký Tài Khoản")
            new_username = st.text_input("Tên đăng nhập mới")
            new_password = st.text_input("Mật khẩu", type="password")
            confirm_password = st.text_input("Xác nhận mật khẩu", type="password")
            full_name = st.text_input("Họ và tên")
            email = st.text_input("Email")
            role = st.selectbox("Vai trò", ["student", "teacher", "admin"])
            
            register = st.form_submit_button("Đăng Ký", type="secondary")
            
            if register:
                if all([new_username, new_password, confirm_password, full_name, email]):
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            if register_user(new_username, new_password, full_name, email, role):
                                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                            else:
                                st.error("Tên đăng nhập đã tồn tại!")
                        else:
                            st.error("Mật khẩu phải có ít nhất 6 ký tự!")
                    else:
                        st.error("Mật khẩu xác nhận không khớp!")
                else:
                    st.warning("Vui lòng điền đầy đủ thông tin!")

# Main application
def show_main_app():
    """Hiển thị ứng dụng chính"""
    # Header với thông tin user
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="main-header">
            <h1>⚖️ Chatbot Tư Vấn - ĐH Luật TPHCM</h1>
            <p>Xin chào <strong>{st.session_state.full_name}</strong> ({st.session_state.role.title()})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Đăng Xuất", type="secondary"):
            for key in ['authenticated', 'user_id', 'username', 'full_name', 'role']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Cấu Hình Hệ Thống")
        
        # Thông tin user
        st.markdown(f"""
        <div class="sidebar-info">
            <h4>👤 Thông tin người dùng</h4>
            <p><strong>Tên:</strong> {st.session_state.full_name}</p>
            <p><strong>Vai trò:</strong> {st.session_state.role.title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload file (chỉ admin và teacher)
        if st.session_state.role in ['admin', 'teacher']:
            st.subheader("📁 Tải Lên Tài Liệu")
            uploaded_files = st.file_uploader(
                "Tải lên tài liệu hướng dẫn, quy chế...",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                help="Hỗ trợ file PDF, Word và Text"
            )
            
            if uploaded_files and st.button("🔄 Xử Lý Tài Liệu", type="primary"):
                with st.spinner("Đang xử lý tài liệu..."):
                    documents, processed_files, error_files = process_uploaded_files_enhanced(uploaded_files)
                    
                    if documents:
                        st.session_state.vector_store = create_enhanced_vector_store(documents)
                        
                        if st.session_state.vector_store:
                            st.markdown(f"""
                            <div class="status-success">
                                ✅ Xử lý thành công {len(processed_files)} file!<br>
                                📊 Tổng số chunks: {st.session_state.vector_store.index.ntotal}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-error">❌ Không thể tạo vector store</div>', unsafe_allow_html=True)
                    
                    if error_files:
                        st.warning("Một số file gặp lỗi:")
                        for error in error_files:
                            st.write(f"• {error}")

        st.divider()
        
        # Lựa chọn LLM
        st.subheader("🤖 Mô Hình AI")
        llm_option = st.selectbox("Chọn mô hình:", ["Gemini", "Grok"])
        
        # Thống kê
        st.divider()
        st.subheader("📊 Thống Kê")
        
        if 'messages' in st.session_state:
            user_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{user_questions}</h3>
                <p>Câu Hỏi Hôm Nay</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.get('vector_store'):
            chunks = st.session_state.vector_store.index.ntotal
            st.markdown(f"""
            <div class="metric-card">
                <h3>{chunks}</h3>
                <p>Tài Liệu Đã Tải</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Lịch sử chat
        if st.button("📜 Xem Lịch Sử Chat"):
            show_chat_history()

    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Kiểm tra API keys
    if llm_option == "Gemini" and not gemini_api_key:
        st.error("⚠️ Vui lòng cung cấp GEMINI_API_KEY trong file .env")
        st.stop()
    elif llm_option == "Grok" and not grok_api_key:
        st.error("⚠️ Vui lòng cung cấp GROK_API_KEY trong file .env")
        st.stop()

    # Khởi tạo LLM
    if llm_option == "Gemini":
        llm = get_enhanced_gemini_llm()
    else:
        llm = get_enhanced_grok_llm()
    
    if not llm:
        st.error("❌ Không thể khởi tạo mô hình AI. Vui lòng kiểm tra API key.")
        st.stop()

    # Khởi tạo chain nếu có vector store
    chain = None
    if st.session_state.vector_store:
        chain = create_enhanced_conversational_chain(st.session_state.vector_store, llm)

    # Hướng dẫn sử dụng ban đầu
    if not st.session_state.messages:
        welcome_message = f"""
        👋 **Chào mừng {st.session_state.full_name} đến với Chatbot Tư Vấn ĐH Luật TPHCM!**
        
        🎯 **Tôi có thể hỗ trợ bạn về:**
        - 📝 **Tuyển sinh**: Thông tin đăng ký, hồ sơ, điểm chuẩn
        - 💰 **Học phí**: Chi phí học tập, học bổng, miễn giảm
        - 📚 **Chương trình đào tạo**: Môn học, tín chỉ, giảng viên
        - 🎭 **Hoạt động sinh viên**: CLB, sự kiện, tình nguyện
        - 🏠 **Hỗ trợ sinh viên**: Ký túc xá, thư viện, cơ sở vật chất
        - 🎓 **Tốt nghiệp**: Thực tập, việc làm, bằng cấp
        
        💡 **Gợi ý câu hỏi:**
        - "Điều kiện xét tuyển vào ngành Luật như thế nào?"
        - "Học phí một năm của trường là bao nhiêu?"
        - "Có những câu lạc bộ nào trong trường?"
        
        🔍 **Lưu ý**: Hãy đặt câu hỏi cụ thể để nhận được hỗ trợ tốt nhất!
        """
        
        st.markdown(welcome_message)

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Hiển thị metadata nếu có
            if "metadata" in message:
                metadata = message["metadata"]
                if metadata.get("category"):
                    st.caption(f"🏷️ Danh mục: {metadata['category']}")
                if metadata.get("sources"):
                    with st.expander("📚 Nguồn tham khảo"):
                        for source in metadata["sources"]:
                            st.write(f"• {source}")

    # Nhập câu hỏi
    if prompt := st.chat_input("💬 Đặt câu hỏi của bạn..."):
        # Hiển thị câu hỏi
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Phân loại câu hỏi
        question_category = classify_question_advanced(prompt)

        # Xử lý và trả lời
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang phân tích và tìm kiếm thông tin..."):
                try:
                    sources = []
                    
                    if chain and st.session_state.vector_store:
                        # Sử dụng RAG với documents
                        response = chain({"question": prompt})
                        answer = response["answer"]
                        
                        # Xử lý nguồn tham khảo
                        if response.get("source_documents"):
                            sources = []
                            for doc in response["source_documents"][:3]:
                                source_info = f"{doc.metadata.get('source_file', 'N/A')} - {doc.metadata.get('file_type', 'Unknown')}"
                                if source_info not in sources:
                                    sources.append(source_info)
                            
                            # Hiển thị nguồn tham khảo
                            if sources:
                                st.markdown("---")
                                with st.expander("📚 Nguồn tham khảo"):
                                    for i, source in enumerate(sources, 1):
                                        st.write(f"**{i}.** {source}")
                                    
                                    # Hiển thị đoạn văn tham khảo
                                    st.markdown("**Đoạn văn tham khảo:**")
                                    for i, doc in enumerate(response["source_documents"][:2], 1):
                                        st.write(f"*Nguồn {i}:* {doc.page_content[:300]}...")
                    else:
                        # Sử dụng LLM trực tiếp
                        enhanced_prompt = f"""
                        Bạn là tư vấn viên chuyên nghiệp của Trường Đại học Luật TPHCM.
                        
                        THÔNG TIN TRƯỜNG:
                        - Tên: Đại học Luật Thành phố Hồ Chí Minh
                        - Địa chỉ: 2 Nguyễn Tất Thành, P.12, Q.4, TPHCM
                        - Website: http://hcmulaw.edu.vn/
                        - Hotline: (028) 39 400 989
                        
                        Danh mục câu hỏi: {question_category}
                        Câu hỏi: {prompt}
                        
                        Hãy trả lời một cách chuyên nghiệp, thân thiện và hữu ích.
                        Sử dụng emoji phù hợp và cấu trúc rõ ràng.
                        Nếu không có thông tin cụ thể, hãy hướng dẫn liên hệ phòng ban phù hợp.
                        """
                        
                        if isinstance(llm, GoogleGenerativeAI):
                            answer = llm.invoke(enhanced_prompt)
                        else:
                            answer = llm.invoke(enhanced_prompt)
                    
                    # Thêm thông tin danh mục
                    if question_category != "Tổng quát":
                        answer = f"**📌 Danh mục: {question_category}**\n\n{answer}"
                    
                    # Thêm footer với thông tin liên hệ
                    answer += f"""
                    
                    ---
                    📞 **Cần hỗ trợ thêm?**
                    - **Hotline**: (028) 39 400 989
                    - **Website**: http://hcmulaw.edu.vn/
                    - **Địa chỉ**: 2 Nguyễn Tất Thành, P.12, Q.4, TPHCM
                    """
                    
                    st.markdown(answer)
                    
                    # Lưu vào database
                    if hasattr(st.session_state, 'user_id'):
                        save_chat_to_db(st.session_state.user_id, prompt, answer, question_category)
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    error_msg = f"""
                    😔 **Xin lỗi, tôi gặp sự cố kỹ thuật!**
                    
                    Vui lòng:
                    - Thử lại câu hỏi sau ít phút
                    - Hoặc liên hệ trực tiếp với phòng tư vấn
                    
                    📞 **Hotline hỗ trợ**: (028) 39 400 989
                    
                    *Mã lỗi: {str(e)[:100]}*
                    """
                    st.error(error_msg)
                    answer = error_msg

        # Lưu tin nhắn với metadata
        message_metadata = {
            "category": question_category,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "metadata": message_metadata
        })

    # Các nút chức năng bổ sung
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Làm mới cuộc trò chuyện"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📥 Xuất lịch sử chat"):
            export_chat_history()
    
    with col3:
        if st.button("🎯 Câu hỏi gợi ý"):
            show_suggested_questions()
    
    with col4:
        if st.button("📊 Thống kê chi tiết"):
            show_detailed_stats()

def show_chat_history():
    """Hiển thị lịch sử chat của user"""
    if hasattr(st.session_state, 'user_id'):
        history = get_user_chat_history(st.session_state.user_id)
        
        if history:
            st.subheader("📜 Lịch Sử Chat")
            for i, (question, answer, category, timestamp) in enumerate(history[:10]):
                with st.expander(f"💬 {question[:50]}... - {timestamp[:10]}"):
                    st.write(f"**🏷️ Danh mục:** {category}")
                    st.write(f"**❓ Câu hỏi:** {question}")
                    st.write(f"**💡 Trả lời:** {answer[:200]}...")
                    st.write(f"**⏰ Thời gian:** {timestamp}")
        else:
            st.info("Chưa có lịch sử chat nào.")

def export_chat_history():
    """Xuất lịch sử chat ra file"""
    if st.session_state.messages:
        chat_data = []
        for msg in st.session_state.messages:
            chat_data.append({
                'Role': msg['role'],
                'Content': msg['content'],
                'Timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(chat_data)
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📥 Tải xuống lịch sử chat",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_suggested_questions():
    """Hiển thị câu hỏi gợi ý"""
    st.subheader("🎯 Câu Hỏi Gợi Ý")
    
    suggestions = {
        "Tuyển sinh": [
            "Điều kiện xét tuyển vào ngành Luật như thế nào?",
            "Hồ sơ đăng ký xét tuyển gồm những gì?",
            "Điểm chuẩn năm ngoái của các ngành là bao nhiêu?",
            "Có những phương thức xét tuyển nào?"
        ],
        "Học phí": [
            "Học phí một năm của trường là bao nhiêu?",
            "Có chính sách miễn giảm học phí không?",
            "Làm thế nào để xin học bổng?",
            "Có thể trả học phí theo đợt không?"
        ],
        "Sinh hoạt": [
            "Có những câu lạc bộ nào trong trường?",
            "Hoạt động ngoại khóa có gì thú vị?",
            "Làm sao để tham gia đoàn thanh niên?",
            "Có chương trình tình nguyện nào không?"
        ]
    }
    
    for category, questions in suggestions.items():
        st.write(f"**{category}:**")
        for q in questions:
            if st.button(q, key=f"suggest_{hash(q)}"):
                st.session_state.suggested_question = q
                st.rerun()

def show_detailed_stats():
    """Hiển thị thống kê chi tiết"""
    st.subheader("📊 Thống Kê Chi Tiết")
    
    if hasattr(st.session_state, 'user_id'):
        history = get_user_chat_history(st.session_state.user_id, 100)
        
        if history:
            # Thống kê theo danh mục
            categories = [h[2] for h in history if h[2]]
            if categories:
                category_counts = pd.Series(categories).value_counts()
                
                st.write("**Phân bố câu hỏi theo danh mục:**")
                for cat, count in category_counts.items():
                    st.write(f"• {cat}: {count} câu hỏi")
            
            # Thống kê theo thời gian
            dates = [h[3][:10] for h in history]
            if dates:
                date_counts = pd.Series(dates).value_counts().sort_index()
                st.write("**Hoạt động theo ngày:**")
                for date, count in date_counts.head(7).items():
                    st.write(f"• {date}: {count} câu hỏi")

# Main function
def main():
    """Hàm main điều khiển luồng ứng dụng"""
    
    # Kiểm tra authentication
    if not st.session_state.get('authenticated', False):
        show_login_page()
    else:
        show_main_app()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
        🏛️ <strong>Trường Đại học Luật Thành phố Hồ Chí Minh</strong><br>
        📍 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM<br>
        📞 Hotline: (028) 39 400 989 | 🌐 Website: http://hcmulaw.edu.vn/<br>
        🤖 <em>Được hỗ trợ bởi AI - Phiên bản nâng cao với xác thực người dùng</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()