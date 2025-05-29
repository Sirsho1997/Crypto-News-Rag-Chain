import os
import json
from dotenv import load_dotenv
from gnews import GNews
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema import Document


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

def fetch_news(cryptos):
    """
    Fetch news articles for a list of cryptocurrencies using the GNews API.

    Args:
        cryptos (List[str]): A list of cryptocurrency names or tickers to search for in the news.

    Returns:
        pd.DataFrame: A combined DataFrame containing news articles ('title', 'description', 'published date', 'url', 'publisher',
       'publisher_title', 'ticker')
    """
    google_news = GNews()
    all_news = []

    for crypto in cryptos:
        try:
            news = google_news.get_news(crypto)
            if len(news) != 0:

                news_df = pd.DataFrame(news)
                news_df['published date'] = pd.to_datetime(
                    news_df['published date'],
                    format='%a, %d %b %Y %H:%M:%S %Z',
                    errors='coerce'
                )
                news_df['publisher_title'] = news_df['publisher'].apply(
                    lambda x: x.get('title') if isinstance(x, dict) else None
                )
                news_df['ticker'] = crypto
                all_news.append(news_df)
                print(f"Done downloading news for {crypto} ")
        except Exception as e:
            print(f"Exception occurred while processing {crypto} : {e}")

    # create a dataframe combining all the news
    combined_news_df = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame([])
    return combined_news_df


def update_news_history(new_df, path):
    """
    Append new news data to a csv file containing historical data.

    Args:
        new_df (pd.DataFrame): A DataFrame containing the latest news data to append.
        path (str): File path to the historical CSV file.

    Returns:
        pd.DataFrame: The updated DataFrame containing both old and new news data.
    """
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        existing_df['published date'] = pd.to_datetime(existing_df['published date'], errors='coerce')
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(by='published date', ascending=False)
    combined_df.to_csv(path, index=False)
    return combined_df


def build_documents(df):
    """
    Convert a pandas DataFrame containing news into a list of LangChain Document objects.

    Each row embedding --> title + description (separated by newlines)

    Metadata is extracted from other columns.

    Args:
        df (pd.DataFrame): A DataFrame containing columns such as 'title', 'description',
                           'content', 'url', 'published date', 'publisher_title', and 'ticker'.

    Returns:
        List[Document]: A list of LangChain Document objects with populated content and metadata.
    """
    documents = []

    for idx, row in df.iterrows():
        content = f"{row['title']} \n {row.get('description')}"
        metadata = {
            "source": row.get("url"),
            "date": str(row.get("published date")),
            "publisher": row.get("publisher_title"),
            "title": row.get("title"),
            "description": row.get("description"),
            "ticker": row.get("ticker"),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def embed_and_store_documents(docs, path):
    """
    Embed and store the news documents in a Chroma vector database using OpenAIEmbeddings.

    Args:
        docs (List[Document]): A list of LangChain Document objects containing news to embed and store.
        path (str): Path to the directory where the Chroma vector database will be saved.

    Returns:
        None
    """

    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(docs, embedding=embeddings, persist_directory=path)
    print(f"Stored {len(docs)} news in Chroma vector DB.")


def main():

    # Access the cryptocurrencies list from asset.json

    with open("asset.json", "r") as f:
        asset = json.load(f)

    cryptocurrencies = asset["cryptocurrencies"]

    path_name = 'google_news'

    # Create the directory if it doesn't exist
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    history_path = f"{path_name}/USDT.csv"
    vector_db_path = "vectordb"

    print("1. Initiating news fetch...")
    news_df = fetch_news(cryptocurrencies)

    if len(news_df) != 0:

        print("2. Updating news history...")
        full_df = update_news_history(news_df, history_path)

        print("3. Building documents...")
        documents = build_documents(full_df)

        print("4. Storing in the vector DB...")
        embed_and_store_documents(documents, vector_db_path)
    else:
        print("No news found for any crypto.")


if __name__ == "__main__":
    main()
