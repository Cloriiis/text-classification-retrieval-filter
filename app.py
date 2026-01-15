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

# --- 3. UI/UX 深度定制 (核心修改区域) ---
st.markdown("""
<style>
    /* === 全局与容器设置 === */
    .stApp {
        background-color: #F4F8FB; /* 更柔和的灰蓝色背景 */
    }
    
    /* === 侧边栏样式 === */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* 1. Navigator 标题方框 */
    .nav-box {
        background-color: #F7FAFC;
        border: 2px solid #3182CE;
        color: #3182CE;
        padding: 12px;
        text-align: center;
        font-weight: 800;
        font-family: 'Arial', sans-serif;
        border-radius: 6px;
        margin-bottom: 30px;
        letter-spacing: 1px;
        box-shadow: 0 2px 4px rgba(49, 130, 206, 0.1);
    }

    /* === 2. 侧边栏导航按钮 (Radio 改造成的方块) === */
    
    /* 核心：去除默认样式 */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0 !important;
        background: transparent !important;
        margin-bottom: 8px !important;
    }
    
    /* 隐藏圆圈 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label input {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* 按钮容器 - 未选中状态 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #718096 !important;
        border-radius: 6px;
        padding: 12px 0 !important;
        width: 100% !important; /* 强制填满宽度 */
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    /* 悬停效果 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #3182CE !important;
        color: #3182CE !important;
        background-color: #F0F7FF !important;
        cursor: pointer;
    }

    /* === 选中状态 (高亮) === */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3182CE !important; /* 深蓝色背景 */
        color: white !important; /* 白色文字 */
        border: 1px solid #3182CE !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(49, 130, 206, 0.3);
        transform: translateY(-1px);
    }
    
    /* 修复文字在方框内的对齐 */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] {
        width: 100%;
        text-align: center;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        margin: 0;
        font-size: 14px;
    }

    /* === 3. 搜索按钮美化 === */
    
    /* 定制 Streamlit 按钮 */
    div.stButton > button {
        background: linear-gradient(90deg, #3182CE 0%, #2B6CB0 100%);
        color: white;
        border: none;
        border-radius: 6px;
        height: 48px; /* 强制高度与输入框一致 */
        font-weight: 600;
        width: 100%;
        margin-top: 1px; /* 微调垂直对齐 */
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2B6CB0 0%, #2C5282 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
        box-shadow: none;
    }

    /* 4. 统计卡片微调 */
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    
    /* 5. 结果卡片 */
    .result-item {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3182CE; /* 左侧蓝色条装饰 */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑 (保持不变) ---
@st.cache_resource
def initialize_system():
    # 模拟逻辑：如果 docs 文件夹为空，您可以手动放入一些 txt 文件测试
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    if not os.path.exists('docs/'):
        os.makedirs('docs/')
    
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_docs = loader.load()
    
    if not raw_docs:
        return None, None, []

    categorized_docs = []
    # 简单的关键词分类逻辑
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

# --- 6. 侧边栏 (UI 更新) ---
with st.sidebar:
    # Navigator 标题方框
    st.markdown('<div class="nav-box">NAVIGATOR</div>', unsafe_allow_html=True)
    
    # 导航选项 (纯文本)
    nav_options = ["ALL ARCHIVES"] + category_list
    
    # 这里的 key 很重要，确保状态同步
    selected_option = st.radio(
        "Navigation", 
        nav_options, 
        label_visibility="collapsed"
    )
    
    selected_category = selected_option

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 统计信息
    col1, col2 = st.columns(2)
    
    total_count = len(raw_docs) if raw_docs else 0
    current_count = "All"
    if selected_category != "ALL ARCHIVES" and raw_docs:
        current_count = sum(1 for d in raw_docs if d.metadata.get('category') == selected_category)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:20px; font-weight:bold; color:#2D3748;">{total_count}</div>
            <div style="font-size:10px; color:#718096;">TOTAL</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:20px; font-weight:bold; color:#3182CE;">{current_count}</div>
            <div style="font-size:10px; color:#718096;">CURRENT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("System v3.2 | High Contrast UI")

# --- 7. 主界面 (搜索栏优化) ---
st.markdown("## 🔎 Information Retrieval")
st.markdown("检索存档中的专业资讯与文档")

st.markdown("<br>", unsafe_allow_html=True)

# 使用 columns 布局，vertical_alignment="bottom" 确保按钮和输入框底部对齐
search_col1, search_col2 = st.columns([5, 1], vertical_alignment="bottom")

with search_col1:
    # 搜索框
    query = st.text_input("Search Query", placeholder="输入关键词...", label_visibility="collapsed")

with search_col2:
    # 搜索按钮 - CSS 已经将其高度设为 48px 以匹配输入框
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

            # 结果卡片样式优化
            st.markdown(f"""
            <div class="result-item">
                <div style="font-size:1.1rem; font-weight:bold; color:#2B6CB0; margin-bottom:5px;">
                    📄 {file_name}
                </div>
                <div style="margin-bottom:12px;">
                    <span style="background:#EBF8FF; color:#2C5282; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;">{cat_tag}</span>
                </div>
                <div style="color:#4A5568; font-size:14px; line-height:1.6;">
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