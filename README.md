# Crypto-News-Rag-Chain


Crypto Rag is an AI-powered crypto news analysis tool that uses **retrieval-augmented generation (RAG)**, **LLM models**, and **LangChain** to track, store, embed, and query the latest cryptocurrency news.

## Features

- **Fetch Crypto News**  
  Pulls live cryptocurrency news articles using the [GNews API]([https://polymarket.com/](https://github.com/ranahaani/GNews/)) for assets defined in asset.json.

- **Embed with OpenAI & Store in Chroma**  
 Converts news articles into [LangChain](https://www.langchain.com/) Document objects, then embeds them using [OpenAI](https://openai.com/) and stores them in the [Chroma](https://www.trychroma.com/) vector database locally.


- **Query via RAG-Powered QA Chain**  
  Chat with an LLM agent that retrieves relevant crypto news from the vector database to answer queries, along withthe  source.

##  Project Structure

```
├── fetch_and_embed.py        # Main script to fetch, process, and embed crypto news
├── query.py                  # Conversational QA chain over embedded news
├── asset.json                # List of cryptocurrencies to monitor
├── google_news/USDT.csv      # Persisted historical news data 
├── vectordb/                 # Persisted Chroma vector DB directory
├── .env                         
├── requirements.txt             
└── README.md

```

##  Quickstart

### 1. Clone the Repo

```bash

git clone https://github.com/Sirsho1997/Crypto-News-Rag-Chain.git
cd Crypto-News-Rag-Chain
```

### 2. Install Requirements
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
```

### 4. Run the News Fetcher and generate Embeddings
```bash
python fetch_and_embed.py 
```

### 5.  Chat with the QA Bot
```bash
python query.py
```

## Example Conversation

<img width="687" alt="image" src="https://github.com/user-attachments/assets/209e932b-5985-4100-ade2-f49247980862" />


## Acknowledgements
[GNews API]([https://polymarket.com/](https://github.com/ranahaani/GNews/)) —  lightweight Python Package that provides an API to search for articles on Google News and returns a usable JSON response!
[Chroma](https://www.trychroma.com/) - Fast, persistent vector storage
[LangChain](https://www.langchain.com/) — Framework for LLM pipelines and RAG systems
