from agent.llm import llm_chat_deepseek as llm
print(llm)
response = llm.invoke("test")
print(response)