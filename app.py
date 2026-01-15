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

# --- 2. 页面设置 (去 Emoji，改用专业图标) ---
st.set_page_config(
    page_title="InfoStream - 专业资讯归档系统",
    page_icon="📑",  # 仅保留标题栏一个图标
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS 深度定制 (去卡片化，走专业文档风) ---
st.markdown("""
<style>
    /* 全局字体与背景 - 更加冷淡严谨 */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* 侧边栏样式重置 */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 标题样式 - 深色衬线体 */
    h1, h2, h3 {
        color: #262730;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 搜索结果列表项样式 (替代之前的 Card) */
    .result-item {
        padding: 15px 0;
        border-bottom: 1px solid #E6E6E6;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A73E8; /* Google Link Blue */
        margin-bottom: 5px;
    }
    .result-meta {
        font-size: 0.85rem;
        color: #5F6368;
        font-family: monospace;
        margin-bottom: 8px;
    }
    .result-snippet {
        font-size: 0.95rem;
        color: #3C4043;
        line-height: 1.5;
    }
    
    /* 隐藏 Streamlit 默认的按钮边框，让界面更干净 */
    div.stButton > button {
        border-radius: 4px;
        background-color: #008080; /* Teal Color */
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #006666;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑：自动分类与索引 ---
@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 读取 data 目录下所有文件
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    # --- 关键修改：自动打标签 ---
    categorized_docs = []
    
    # 关键词定义 (保持你的新分类)
    ai_keywords = ['learning', 'neural', 'intelligence', 'gpt', 'python', 'data', 'cloud']
    fintech_keywords = ['blockchain', 'bitcoin', 'payment', 'finance', 'wallet', 'economy', 'bank']
    humanities_keywords = ['history', 'culture', 'art', 'philosophy', 'literature', 'civilization', 'museum']
    
    for doc in raw_docs:
        filename = doc.metadata['source'].lower()
        content = doc.page_content.lower()
        
        # 默认分类 (无 Emoji)
        category = "General / Uncategorized"
        
        # 根据文件名或内容判断分类 (移除 Emoji)
        if any(k in filename or k in content for k in ai_keywords):
            category = "AI & Technology"
        elif any(k in filename or k in content for k in fintech_keywords):
            category = "FinTech & Economy"
        elif any(k in filename or k in content for k in humanities_keywords):
            category = "Humanities & History"
            
        doc.metadata['category'] = category
        categorized_docs.append(doc)

    # 强制定义分类列表顺序 (解决分类显示不全的问题)
    # 即使文件夹里没有文件，这些选项也会显示，保证 UI 结构完整
    fixed_categories = ["AI & Technology", "FinTech & Economy", "Humanities & History", "General / Uncategorized"]

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(categorized_docs)
    
    # 建立向量索引
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, raw_docs, fixed_categories

# --- 5. 初始化 ---
with st.spinner("Initializing Archives..."):
    vector_db, raw_docs, category_list = initialize_system()

# --- 6. 侧边栏：控制面板风格 ---
with st.sidebar:
    st.markdown("### 🗂️ Document Navigator")
    
    # 使用 Radio 组件但样式更简洁
    selected_category = st.radio(
        "Select Category:",
        ["ALL ARCHIVES"] + category_list
    )
    
    st.markdown("---")
    
    # 仪表盘式的数据展示
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Docs", value=len(raw_docs))
    with col2:
        if selected_category != "ALL ARCHIVES":
            count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)
            st.metric(label="Current", value=count)
        else:
            st.metric(label="Current", value="All")

    st.markdown("---")
    st.caption("System v2.0 | Topic 2 Classification Build")

# --- 7. 主界面：搜索引擎风格 ---

st.markdown("## 🔎 Information Retrieval System")
st.markdown("Type keywords to search across the categorized database.")

# 搜索栏布局：更像 Google
search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")

with search_col1:
    query = st.text_input("Search Query", placeholder="e.g., impact of blockchain", label_visibility="collapsed")
with search_col2:
    search_btn = st.button("Search", use_container_width=True)

st.markdown("---")

# --- 8. 检索与结果展示 (列表式，非卡片式) ---
if (query or search_btn) and vector_db:
    start_time = time.time()
    
    # 1. 宽泛召回
    results = vector_db.similarity_search(query, k=20)
    
    # 2. 严格过滤
    if selected_category != "ALL ARCHIVES":
        filtered_results = [doc for doc in results if doc.metadata.get('category') == selected_category]
    else:
        filtered_results = results

    # 取 Top 5
    final_results = filtered_results[:5]

    # 显示结果头
    if not final_results:
        st.warning(f"No results found in category: {selected_category}")
    else:
        st.markdown(f"**Found {len(final_results)} relevant documents** ({time.time() - start_time:.4f}s)")
        
        for doc in final_results:
            cat_tag = doc.metadata.get('category')
            file_name = doc.metadata['source'].split('/')[-1]
            
            # 使用 HTML 构建“谷歌学术”风格的列表
            st.markdown(f"""
            <div class="result-item">
                <div class="result-title">📄 {file_name}</div>
                <div class="result-meta">
                    <span style="background-color: #E0F2F1; color: #00695C; padding: 2px 6px; border-radius: 4px;">{cat_tag}</span>
                    &nbsp; • &nbsp; Relevance Match
                </div>
                <div class="result-snippet">
                    ...{doc.page_content}...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 使用原生 expander 查看全文 (折叠起来保持干净)
            with st.expander("View Full Context"):
                st.text(doc.page_content)

elif not vector_db:
    st.error("Database Error: Please check data directory.")
elif not query:
    st.info("Awaiting input... Select a category from the sidebar to browse.")