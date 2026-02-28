from PyPDF2 import PdfReader
from langchain_community.callbacks.manager import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
from logging import Logger

# LCEL 相关组件
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.output_parsers import StrOutputParser   # 字符串输出解析器

from agent.llm import llm_chat_deepseek as llm
from agent.embedding_model_langchain import embeddings

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码

    参数:
        pdf: PDF文件对象

    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            Logger.warning(f"No text found on page {page_number}.")

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int]) -> FAISS:
    """
    处理文本并创建向量存储

    参数:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表

    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    # Logger.debug(f"Text split into {len(chunks)} chunks.")
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 从文本块创建知识库
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    # Logger.info("Knowledge base created from text chunks.")
    print("已从文本块创建知识库。")

    # 存储每个文本块对应的页码信息
    knowledge_base.page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}

    return knowledge_base


# 读取PDF文件
pdf_reader = PdfReader('浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_numbers(pdf_reader)

print(f"提取的文本长度: {len(text)} 个字符。")

# 处理文本并创建知识库
knowledgeBase = process_text_with_splitter(text, page_numbers)

# 设置查询问题
query = "客户经理被投诉了，投诉一次扣多少分"
# query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query)

    # 定义 Prompt 模板（与原来 stuff 链的提示类似）
    prompt = ChatPromptTemplate.from_template(
        """根据以下上下文来回答问题。如果你不知道答案，就直接说不知道，不要试图编造。

上下文：
{context}

问题：{question}

答案："""
    )


    # 定义一个函数，将文档列表格式化为单一字符串（上下文）
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])


    # 使用 LCEL 构建问答链
    chain = (
            {
                "context": lambda x: format_docs(x["docs"]),  # 格式化文档作为上下文
                "question": lambda x: x["question"]  # 直接传递问题
            }
            | prompt  # 填充提示模板
            | llm  # 调用语言模型
            | StrOutputParser()  # 解析输出为字符串
    )

    # 准备输入数据
    input_data = {"question": query, "docs": docs}

    # 使用回调函数跟踪 API 调用成本（保持不变）
    with get_openai_callback() as cost:
        # 执行链
        answer = chain.invoke(input_data)  # 直接得到答案字符串
        print(f"查询已处理。成本: {cost}")
        print(answer)  # 输出答案
        print("来源:")

    # 打印来源页码（保持不变）
    unique_pages = set()
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )
        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")