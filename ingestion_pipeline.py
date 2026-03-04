import os
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_document(doc_file = "docs"):
    print(f"loading documents from the folder {doc_file} ....")

    if not os.path.exists(doc_file):
        print("file floder not found")

    loader = DirectoryLoader(
        path = doc_file,
        glob  = "*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
        
    documents = loader.load()

    if len(documents) == 0:
        print("document is empty")

    for i , doc in enumerate(documents):
        print(f"document : {i+1}")
        print(f"metadata : {doc.metadata['source']}")
        print(f"content length : {len(doc.page_content)} characters")
        print(f"content preview : {doc.page_content[:100]}")
        print(f"metadata : {doc.metadata}")

    return documents
 
    
def split_document(documents):  
  
    print("chunking the documents")
     

    test_spliter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = test_spliter.split_documents(documents)

    if chunks:
        for i , chunk in enumerate(chunks[:10]):
            print(f"\n--- Chunk {i+1}")
            print(f"Source : {chunk.metadata['source']}")
            print(f"Lenght : {len(chunk.page_content)} characters")
            print("contents")
            print(chunk.page_content)
            print("-" * 50)

    return chunks

def create_vector_store(chunks,persist_directory = "db/chroma_db"):
    print("creating embedding and storing in chromaDB")

    #embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #THIS MODEL IS A HUGGINFACE

    print("---craeting vector store ---")
    vectorstores = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata={"hnsw:space" : "cosine"}  # this is alogrithem

    )

    print ("--- Finished creating vector store -----") 

    print(f"vector store created and save to {persist_directory}")
    return vectorstores

def main():

    #loading the source file
    documents = load_document(doc_file = "docs")

    # #chuncking  
    chunks = split_document(documents)

    # #Embedding and storing in vector db
    vectorStore = create_vector_store(chunks)

main()

