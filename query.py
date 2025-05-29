import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def initialize_qa_chain(path,  model_name):
    """
    Initialize a Retrieval-based QA chain using a Chroma vector store and OpenAI embeddings.

    Args:
        path (str): Path to the persisted vector database directory.
        model_name (str): The name of the OpenAI language model to used to generate the embedding.

    Returns:
        RetrievalQA: A RetrievalQA chain that can be used for query.
    """

    embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=path, embedding_function=embedding)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model=model_name)
    response = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return response



def print_response(response):
    """
    Return the response answer and its source documents from a QA system response.

    Args:
        response (dict): The response dictionary from the QA chain. Expected to contain:
            - "result" (str): The generated answer.
            - "source_documents" (List[Document]): A list of source documents with metadata.

    Returns:
        None
    """
    print("\n BOT Answer:\n")
    print(response["result"])

    print("\n Sources:")
    for i, doc in enumerate(response["source_documents"], 1):
        metadata = doc.metadata
        print(f"\n[{i}] {metadata.get('source')}")
        print(f" {metadata.get('ticker')} -> Title: {metadata.get('title')}  Publisher: {metadata.get('publisher')}")
        print(f" Date: {metadata.get('date')}")
        print("---")


def main():

    path = "vectordb"
    model_name = "gpt-3.5-turbo"

    qa_chain = initialize_qa_chain(path, model_name)


    while True:
        query = input("\nEnter your crypto news Query\n Type 'quit' to exit!\n")

        if query.lower() == 'quit':
            print("Exiting!")
            break
        response = qa_chain.invoke(query)

        print_response(response)


if __name__ == "__main__":
    main()
