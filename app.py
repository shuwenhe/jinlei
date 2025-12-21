import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 最新版 LangChain 的正确导入
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_chains.retrieval import create_retrieval_chain

# 页面配置
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="🔍",
    layout="wide"
)

# 标题
st.title("🔍 金雷科技 · 六道工序知识库问答系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("📚 系统信息")
    st.info("当前知识库：六道工序.docx")
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. 在下方输入您的问题
    2. 点击“搜索”按钮
    3. 系统将基于知识库生成专业回答，并显示参考文档片段
    """)
    st.markdown("### 模型信息")
    st.caption("Embedding: bge-m3\nLLM: qwen:7b")

# 加载知识库
@st.cache_resource
def load_knowledge_base():
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        index_path = "./faiss_jinlei_index"
        
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.sidebar.success("✅ 知识库加载成功！")
            return vector_store
        else:
            st.sidebar.error(f"❌ 未找到索引文件: {index_path}\n请先运行 build_knowledge_base.py")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ 加载失败: {e}")
        return None

vector_store = load_knowledge_base()

# 输入区域
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        "请输入您的问题：",
        placeholder="例如：六道工序的具体步骤是什么？",
        key="query_input"
    )
with col2:
    st.write("")
    st.write("")
    search_button = st.button("🔍 搜索", type="primary")

# 处理查询
if search_button and query:
    if not query.strip():
        st.warning("⚠️ 请输入问题内容")
    elif vector_store is None:
        st.error("❌ 知识库未加载成功，请检查索引文件和 Ollama 服务")
    else:
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                llm = Ollama(model="qwen:7b", temperature=0.3)
                
                template = """
                你是金雷科技的专业助手，请根据以下参考文档内容，准确、专业地回答用户问题。
                如果参考文档中没有相关信息，请回复：“根据当前知识库，我无法回答这个问题。”

                参考文档：
                {context}

                用户问题：{question}

                回答：
                """
                prompt = PromptTemplate.from_template(template)
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                docs = response["context"]
                
                st.subheader("🤖 智能回答")
                st.markdown(answer)
                
                st.subheader(f"📋 参考文档片段（共 {len(docs)} 条）")
                for i, doc in enumerate(docs):
                    with st.expander(f"📄 片段 {i+1}", expanded=(i == 0)):
                        st.markdown("**内容：**")
                        st.markdown(doc.page_content)
                        if doc.metadata:
                            st.markdown("**元数据：**")
                            for k, v in doc.metadata.items():
                                st.markdown(f"- **{k}:** {v}")
                                
            except Exception as e:
                st.error(f"❌ 处理失败：{e}\n请检查 Ollama 是否运行，并已拉取 qwen:7b 和 bge-m3 模型")

# 底部信息
st.markdown("---")
st.caption("💡 本系统基于 '六道工序.docx' 构建，使用本地 Ollama (qwen:7b + bge-m3) 运行，完全离线隐私安全。")
