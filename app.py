import os
# --- 1. 配置镜像源 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 2. 页面设置 ---
st.set_page_config(
    page_title="InfoStream - 专业资讯归档系统",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS 深度定制 (浅蓝专业风格) ---
st.markdown("""
<style>
    /* 全局背景：极淡的海洋蓝 */
    .stApp {
        background-color: #F0F7FF;
    }
    
    /* 侧边栏：浅天蓝色调 */
    [data-testid="stSidebar"] {
        background-color: #E3EEF9;
        border-right: 1px solid #D1E3F8;
    }
    
    /* 标题样式：深海蓝 */
    h1, h2, h3 {
        color: #1A365D;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* 搜索结果项：白色底色，带微弱蓝色投影 */
    .result-item {
        background-color: #FFFFFF;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 8px;
        border: 1px solid #E1E8F0;
        box-shadow: 0 2px 4px rgba(26, 54, 93, 0.05);
    }
    
    /* 搜索结果标题：更具活力的蓝色 */
    .result-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2B6CB0;
        margin-bottom: 6px;
    }
    
    /* 元数据与标签 */
    .result-meta {
        font-size: 0.85rem;
        color: #718096;
        margin-bottom: 10px;
        font-family: 'SFMono-Regular', monospace;
    }
    
    .cat-tag {
        background-color: #EBF8FF;
        color: #2C5282;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #BEE3F8;
    }
    
    /* 正文摘要 */
    .result-snippet {
        font-size: 0.95rem;
        color: #2D3748;
        line-height: 1.6;
    }
    
    /* 按钮样式：商务蓝色 */
    div.stButton > button {
        border-radius: 6px;
        background-color: #3182CE;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #2B6CB0;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
        transform: translateY(-1px);
    }

    /* 输入框聚焦色 */
    .stTextInput input:focus {
        border-color: #3182CE !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑 ---
@st.cache_resource
def initialize_system():
    # 注意：如果本地没有模型，会自动从镜像下载
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 确保目录存在
    if not os.path.exists('docs/'):
        os.makedirs('docs/')
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    # 自动打标签逻辑
    categorized_docs = []
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        category = "General / Uncategorized"
        
        if any(k in filename or k in content for k in ai_keywords):
            category = "AI & Technology"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "FinTech & Economy"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "Humanities & History"
            
        doc.metadata['category'] = category
        categorized_docs.append(doc)

    fixed_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History", "General / Uncategorized"]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, raw_docs, fixed_categories

# --- 5. 初始化 ---
with st.spinner("Initializing Azure Archive System..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏 ---
with st.sidebar:
    st.markdown("### 🗂️ Navigator")
    selected_category = st.radio("Select Category:", ["ALL ARCHIVES"] + category_list)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total", value=len(raw_docs) if raw_docs else 0)
    with col2:
        if selected_category != "ALL ARCHIVES" and raw_docs:
            count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)
            st.metric(label="Current", value=count)
        else:
            st.metric(label="Current", value="All")
    st.markdown("---")
    st.caption("System v2.1 | Azure Theme")

# --- 7. 主界面 ---
st.markdown("## 🔎 Information Retrieval System")
st.markdown("检索存档中的专业资讯与文档")

search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")
with search_col1:
    query = st.text_input("Search Query", placeholder="输入关键词，例如：人工智能的发展...", label_visibility="collapsed")
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
        st.warning(f"未在 【{selected_category}】 分类中找到相关内容。")
    else:
        st.markdown(f"**找到 {len(final_results)} 条相关记录** (用时 {time.time() - start_time:.4f}s)")
        
        for doc in final_results:
            cat_tag = doc.metadata.get('category')
            file_name = doc.metadata['source'].split('/')[-1]
            full_file_path = doc.metadata['source']
            
            # 查找原文
            full_content = "未找到全文内容"
            for raw_doc in raw_docs:
                if raw_doc.metadata['source'] == full_file_path:
                    full_content = raw_doc.page_content
                    break

            # 蓝色调列表显示
            st.markdown(f"""
            <div class="result-item">
                <div class="result-title">📄 {file_name}</div>
                <div class="result-meta">
                    <span class="cat-tag">{cat_tag}</span>
                    &nbsp; • &nbsp; ⚖️ 相关度匹配
                </div>
                <div class="result-snippet">
                    {doc.page_content}... 
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 查看完整文档"):
                st.markdown(full_content)
                st.caption(f"文件路径: {full_file_path}")

elif not vector_db:
    st.info("请在 docs/ 目录下放入 .txt 文件后启动系统。")
elif not query:
    st.info("💡 提示：在上方搜索框输入内容，或在左侧选择分类浏览。")