from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

from agent.embedding_model_langchain import embeddings
from agent.llm import llm_chat_deepseek as llm

# 加载向量数据库，添加 allow_dangerous_deserialization=True 参数已允许反序列号
vectorstore = FAISS.load_local("./faiss-1", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm= llm
)

# 查询示例
query = "客户经理的考核标准是什么？"
# 执行查询
results = retriever.invoke(query)

# 打印结果
print(f"查询: {query}")
print(f"找到 {len(results)} 个相关文档:")
for i, doc in enumerate(results):
    print(f"\n文档 {i+1}:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)