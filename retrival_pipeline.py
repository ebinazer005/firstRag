from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_groq import ChatGroq

persist_directory= "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #THIS MODEL IS A HUGGINFACE

db = Chroma(
    persist_directory= persist_directory,
    embedding_function = embedding_model,
    collection_metadata= {"hnsw:space" : "cosine"}
)

query = "When is Google Founded i wanna exact data with simple line"

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
# 9. "what year google were intraduce"

retrival = db.as_retriever(search_kwargs = {'k':3})   #k : 3 means it pick top three chunks 

retrival_docs = retrival.invoke(query)

print(f"user query  :{query}")

print("context")
for i, doc in enumerate(retrival_docs ,1):
    print(f"Documents{i}\n{doc.page_content}")

#insearting  to LLM

insert_theInput = f"""based on the following document , please aunswer this question : {query}

Documents : {chr(10).join(f"-{doc.page_content}" for doc in retrival_docs)}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatGroq(model="llama-3.1-8b-instant", api_key="your api key")



message =[
    SystemMessage(content="you are a helpful assistent."),
    HumanMessage(content=insert_theInput)
]

result = model.invoke(message)

print("\n--- Generated Response ---")

print("content only:")
print(result.content)