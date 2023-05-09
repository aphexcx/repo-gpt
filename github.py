import os
import subprocess
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import logging

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
logging.basicConfig(level=logging.INFO)


def clone_repository(repo_url, local_path):
    subprocess.run(["git", "clone", repo_url, local_path])


def is_binary(file_path):
    """Check if the file is binary."""
    with open(file_path, 'rb') as file:
        chunk = file.read(1024)
    return b'\0' in chunk


def load_docs(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if not is_binary(file_path):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    logging.error(f"Error loading file {file}: {str(e)}")
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def main(repo_url, root_dir, deep_lake_path):
    docs = load_docs(root_dir)
    texts = split_docs(docs)
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=deep_lake_path, embedding_function=embeddings)
    db.add_documents(texts)


if __name__ == "__main__":
    repo_url = os.environ.get('REPO_URL')
    root_dir = "/Users/afik_cohen/repos/ppl-ai/clean-android/"
    deep_lake_path = os.environ.get('DEEPLAKE_DATASET_PATH')
    main(repo_url, root_dir, deep_lake_path)
