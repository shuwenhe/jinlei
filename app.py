import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# --- 配置参数 (需与 build_knowledge_base.py 保持一致) ---
FAISS_INDEX_PATH = "./faiss_jinlei_index"
OLLAMA_LLM_MODEL = "qwen:7b" # 确保已拉取 Qwen 模型
OLLAMA_EMBEDDING_MODEL = "m3e-base" 

# --- 初始化核心组件 (使用 st.cache_resource 避免重复加载) ---

@st.cache_resource
def load_rag_chain():
    """加载 LLM、Embedding 模型、FAISS 索引并创建 RAG 问答链。"""
    try:
        # 1. 初始化 Embedding
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url="http://localhost:11434"
        )
        
        # 2. 加载 FAISS 知识库
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error(f"❌ 错误：未找到 FAISS 索引目录 '{FAISS_INDEX_PATH}'。")
            st.error("请先运行 'python build_knowledge_base.py' 构建知识库！")
            return None

        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3}) # 检索最相关的 3 个文档块

        # 3. 初始化 LLM
        llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0.1, base_url="http://localhost:11434")

        # 4. 定制 Prompt 模板 (优化回答结构和角色定位)
        template = """
        你是一名资深的**金雷科技**维修工程师。
        请根据用户的问题和提供的**参考维修文档片段**，给出专业、清晰、分步的维修建议。
        
        **回答要求和结构：**
        1. **故障诊断:** 简要总结用户问题的核心故障点。
        2. **维修建议/步骤:** 列出具体的、可操作的**分步**解决方案。
        3. **参考依据:** 指出建议是基于哪些文档信息得出的。
        
        **【参考维修文档片段】**
        {context}
        
        **【用户提出的问题】**
        {question}
        
        **【金雷科技维修建议】**
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 5. 创建 RAG 链
        document_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        st.success("✅ RAG 系统初始化成功！")
        return retrieval_chain

    except Exception as e:
        st.error(f"❌ RAG 系统初始化失败，请检查 Ollama 服务是否运行，或模型是否拉取: {e}")
        return None

# --- Streamlit Web 界面 ---

st.set_page_config(page_title="金雷科技智能维修问答系统", layout="wide")
st.title("⚡ 金雷科技大模型维修知识问答系统")
st.caption(f"由 Ollama ({OLLAMA_LLM_MODEL} + {OLLAMA_EMBEDDING_MODEL}) & LangChain 提供技术支持")

# 尝试加载 RAG 链
rag_chain = load_rag_chain()

if rag_chain:
    # 初始化历史聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 用户输入
    if prompt := st.chat_input("请输入您遇到的维修问题，例如：'设备运行时，指示灯闪烁但无法启动，应该如何处理？'"):
        # 存储并显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 调用 RAG 链
        with st.spinner("🔧 正在查询知识库并生成专业维修建议..."):
            try:
                # 调用 RAG 链进行问答
                response = rag_chain.invoke({"input": prompt}) 
                
                # LLM 生成的最终回答
                assistant_response = response['answer']
                
                # 检索到的源文档信息
                source_docs = response['context']
                
            except Exception as e:
                assistant_response = f"抱歉，系统在处理请求时发生错误：{e}"
                source_docs = []

            # 显示模型回答
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                
                # 添加文档引用，实现可溯源性 (功能 4 的优化)
                if source_docs:
                    with st.expander("📚 查看参考文档引用"):
                        st.markdown(f"**总共检索到 {len(source_docs)} 条相关文档片段。**")
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get('source', '未知文档')
                            st.subheader(f"片段 {i+1}：来自 {os.path.basename(source_name)}")
                            st.code(doc.page_content[:500] + "...", language='text') # 只显示前 500 字符

            # 存储模型消息
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
# 运行命令: streamlit run app.py
