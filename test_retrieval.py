from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 加载 embedding 模型（与构建时一致）
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 加载 FAISS 索引
INDEX_PATH = "./faiss_jinlei_index"  # 注意你的目录名
db = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True  # 必须加这个参数（新版 LangChain 安全要求）
)

# 测试检索
query = "六道工序是什么？"  # 用你的文档相关问题测试
results = db.similarity_search(query, k=3)  # k=3 返回 top 3 结果

print(f"检索到 {len(results)} 条结果：")
for i, doc in enumerate(results, 1):
    print(f"\n--- 结果 {i} ---")
    print("内容:", doc.page_content)
    print("来源:", doc.metadata.get('source', '未知'))
