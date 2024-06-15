import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

#Creating a RAG application to be able to query the NBA Collective Bargaining Agreement (CBA)
#Need to run the 5_populate_database.py script first to add the CBA PDF to the ChromaDB first
# RAG (retrieval augmentation generation) fine tune a model with new information. Add a new layer of data that can be referenced which uses vector databases. This feeds data into vector data which uses inear alebra to find data that uses similarity scoring to give accurate response.
# chroma DB - run locally
# RAG as a service: https://vectara.com/compare-pinecone-io-vs-vectara/

CHROMA_PATH = "chroma_nba_cba"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



def query_rag():
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #Finish adding the RAG query below
    user_message = input("enter topic to learn about: ")

    results = db.similarity_search_with_score(user_message, k=5)

    context_text = "\n\n----\n\n".join([doc.page_content for doc, _score in results])

    print("context_text:", context_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=user_message)

    model= Ollama(model="llama3")

    response_text = model.invoke(prompt)

    print(response_text)


query_rag()
