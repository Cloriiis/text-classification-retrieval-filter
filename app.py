import os
# --- 1. 配置镜像源 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. 页面设置 ---
st.set_page_config(
    page_title="InfoStream - 专业资讯归档系统",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. UI/UX 深度定制 ---
st.markdown("""
<style>
    /* 1. 全局背景统一 */
    .stApp {
        background-color: #F0F7FF;
    }
    
    header[data-testid="stHeader"] {
        background-color: #F0F7FF;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 2. 侧边栏整体背景 */
    [data-testid="stSidebar"] {
        background-color: #EBF4FF;
        border-right: 1px solid #D6E4F0;
    }
    
    /* === Navigator 标题 (方框、居中) === */
    .nav-header-box {
        background-color: #FFFFFF;
        border: 2px solid #2B6CB0; /* 深蓝色边框 */
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 25px;
        color: #2B6CB0;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* === 侧边栏导航按钮化改造 (方框样式) === */
    
    /* 隐藏原生单选按钮的圆圈输入框 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label input {
        display: none; 
    }
    
    /* 隐藏原生单选按钮圆圈的占位 div (防止左侧留白) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* 选项容器基础样式 (未选中状态 - 白色方框) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E0;
        border-radius: 6px;
        padding: 12px 0px; /* 上下内边距 */
        margin-bottom: 10px;
        transition: all 0.2s ease;
        color: #4A5568;
        font-weight: 600;
        display: flex;
        justify-content: center; /* 文字居中 */
        align-items: center;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* 鼠标悬停效果 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #3182CE;
        color: #3182CE;
        background-color: #F7FAFC;
    }
    
    /* 选中状态 (蓝色背景方框) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3182CE !important;
        color: white !important;
        border-color: #3182CE !important;
        box-shadow: 0 4px 6px rgba(49, 130, 206, 0.3);
    }
    
    /* 调整 Markdown 容器以确保文字完全居中 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        margin: 0; /* 移除文字默认边距 */
        font-size: 0.95rem;
    }

    /* 3. 统计卡片样式 */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2C5282;
    }
    .metric-label {
        font-size: 12px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 4. 搜索结果样式 */
    .result-item {
        background-color: #FFFFFF;
        padding: 24px;
        margin-bottom: 16px;
        border-radius: 12px;
        border: 1px solid #E6F0FA;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.03);
        transition: transform 0.2s;
    }
    .result-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.08);
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2B6CB0;
        margin-bottom: 8px;
    }
    .cat-tag {
        background-color: #EBF8FF;
        color: #2C5282;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    div.stButton > button {
        background-color: #3182CE;
        color: white;
        border-radius: 8px;
        height: 46px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑 ---
@st.cache_resource
def initialize_system():
    # 模拟数据模式，防止报错 (如果此行不需要可删除，保留原始逻辑)
    # 真实环境请确保 docs/ 文件夹存在且有文件
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    if not os.path.exists('docs/'):
        os.makedirs('docs/')
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    categorized_docs = []
    # 关键词定义
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        category = "General"
        
        if any(k in filename or k in content for k in ai_keywords):
            category = "AI & Technology"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "FinTech & Economy"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "Humanities & History"
            
        doc.metadata['category'] = category
        categorized_docs.append(doc)

    display_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History"]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, raw_docs, display_categories

# --- 5. 初始化 ---
with st.spinner("Initializing System..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏 (重构版 - 方框风格) ---
with st.sidebar:
    # 1. 标题改为方框样式
    st.markdown('<div class="nav-header-box">NAVIGATOR</div>', unsafe_allow_html=True)
    
    # 2. 构造纯文字列表（无 Emoji）
    nav_options = ["ALL ARCHIVES"] + category_list
    
    # 3. Radio 组件，CSS 已经将其魔改为方框按钮
    selected_option = st.radio(
        "Navigation", 
        nav_options, 
        label_visibility="collapsed"
    )
    
    # 4. 直接赋值，不需要字符串切片
    selected_category = selected_option

    st.markdown("---")
    
    # 5. 统计卡片
    col1, col2 = st.columns(2)
    
    total_count = len(raw_docs) if raw_docs else 0
    current_count = "All"
    if selected_category != "ALL ARCHIVES" and raw_docs:
        current_count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_count}</div>
            <div class="metric-label">Total Docs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_count}</div>
            <div class="metric-label">Current</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("System v3.1 | Box Style")

# --- 7. 主界面 ---
st.markdown("## 🔎 Information Retrieval")
st.markdown("检索存档中的专业资讯与文档")

search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")
with search_col1:
    query = st.text_input("Search Query", placeholder="输入关键词...", label_visibility="collapsed")
with search_col2:
    search_btn = st.button("Search", use_container_width=True)

st.markdown("---")

# --- 8. 检索与结果展示 ---
if (query or search_btn) and vector_db:
    start_time = time.time()
    results = vector_db.similarity_search(query, k=15)
    
    if selected_category != "ALL ARCHIVES":
        filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
    else:
        filtered_results = results

    final_results = filtered_results[:5]

    if not final_results:
        st.info(f"未在 【{selected_category}】 中找到相关内容。")
    else:
        st.markdown(f"**找到 {len(final_results)} 条相关记录** (用时 {time.time() - start_time:.4f}s)")
        
        for doc in final_results:
            cat_tag = doc.metadata.get('category')
            file_name = doc.metadata['source'].split('/')[-1]
            full_file_path = doc.metadata['source']
            
            full_content = "未找到全文内容"
            for raw_doc in raw_docs:
                if raw_doc.metadata['source'] == full_file_path:
                    full_content = raw_doc.page_content
                    break

            st.markdown(f"""
            <div class="result-item">
                <div class="result-title">📄 {file_name}</div>
                <div style="margin-bottom:10px;">
                    <span class="cat-tag">{cat_tag}</span>
                    <span style="color:#A0AEC0; font-size:0.8rem; margin-left:10px;">相关度匹配</span>
                </div>
                <div style="color:#4A5568; line-height:1.6;">
                    {doc.page_content}... 
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 查看完整文档"):
                st.markdown(full_content)

elif not vector_db:
    st.info("请在 docs/ 目录下放入 .txt 文件后启动系统。")
elif not query:
    st.info("💡 在上方搜索框输入关键词开启检索。")