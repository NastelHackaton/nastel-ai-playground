import uuid
from operator import itemgetter
from langchain_core.documents.base import Document
from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import requests
from requests.auth import HTTPBasicAuth
import zipfile
import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain, SequentialChain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from operator import itemgetter
from collections import defaultdict

# GitHub repository information
owner = ""
repo = ""
branch = ""
token = ""

OPENAPI_API_KEY = ""

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAPI_API_KEY

# OpenAI API key
openai.api_key = OPENAPI_API_KEY

# Qdrant client setup
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "{owner}_{repo}_code_embeddings".format(owner=owner, repo=repo)

# Step 1: Download and unzip the repository
def download_and_extract_repo(owner, repo, branch, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    response = requests.get(url, auth=HTTPBasicAuth(owner, token), stream=True)

    zip_file_path = "repository.zip"
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("repo_files/{owner}_{repo}".format(owner=owner, repo=repo))

    print("Repository downloaded and extracted.")

# Step 2: Generate OpenAI embeddings for file content
def get_openai_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def get_file_language(file_path):
    """
    Determine the programming language of a file based on its extension.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    language_map = {
        '.php': 'PHP',
        '.js': 'JavaScript',
        '.py': 'Python',
        '.java': 'Java',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.ts': 'TypeScript',
        '.cs': 'C#',
        '.cpp': 'C++',
        '.h': 'C/C++ Header',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sql': 'SQL',
        '.xml': 'XML',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML'
    }

    return language_map.get(extension, 'Unknown')

def save_embedding_to_qdrant(file_name, chunk, embedding, point_id):
    file_path = os.path.join("repo_files", file_name)
    language = get_file_language(file_path)

    point = PointStruct(
        id=str(point_id),
        vector=embedding,
        payload={
            "file_name": file_name,
            "chunk": chunk,
            "language": language
        }
    )
    qdrant_client.upsert(collection_name=collection_name, points=[point])
    print(f"Saved embedding for {file_name} ({language}) with point ID {point_id}.")


def is_useful_file(file_path):
    """
    Determine if a file is useful for code analysis based on its extension and path.
    """

    # List of file/directory patterns to exclude
    exclude_patterns = {
        'vendor/', 'node_modules/', 'test/', 'tests/', '.git/',
    }

    # Check if the file should be excluded based on patterns
    for pattern in exclude_patterns:
        if pattern in file_path:
            return False

    return True

def process_files_and_save_embeddings():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8191, chunk_overlap=200)

    for root, _, files in os.walk("repo_files/{owner}_{repo}".format(owner=owner, repo=repo), topdown=True):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if not is_useful_file(file_path):
                print(f"Skipping non-useful file: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    file_content = file.read()

                chunks = text_splitter.split_text(file_content)

                for chunk in chunks:
                    embedding = get_openai_embedding(chunk)
                    save_embedding_to_qdrant(file_path, chunk, embedding, uuid.uuid4())

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Initialize the Qdrant collection
def initialize_qdrant():
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # OpenAI embeddings are 1536-dimensional
    )
    print(f"Qdrant collection '{collection_name}' initialized.")

def semantic_chunking(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def query_expansion(query, llm):
    prompt = ChatPromptTemplate.from_template(
        "Generate three different versions of the following query to retrieve relevant information:\n{query}"
    )
    chain = prompt | llm | StrOutputParser()
    expanded_queries = chain.invoke({"query": query}).split("\n")
    return [query] + expanded_queries


def rerank_documents(documents, query, llm, top_n=5):
    prompt = ChatPromptTemplate.from_template(
        "Rate the relevance of the following document to the query '{query}' on a scale of 1-10:\n{document}\nRelevance score:"
    )
    chain = prompt | llm | StrOutputParser()

    scored_docs = []
    for doc in documents:
        score_str = chain.invoke({"query": query, "document": doc.page_content})
        try:
            score = float(score_str)
        except ValueError:
            # If the score is not a valid float, assign a default low relevance score
            score = 0.0
        scored_docs.append((doc, score))

    return [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_n]]

def format_docs(docs):
    return "\n\n".join(
        f"File: {doc.metadata.get('file_name', 'Unknown')} (Language: {doc.metadata.get('language', 'Unknown')})\n Content: {doc.page_content}"
        for doc in docs
    )

def get_all_documents_from_qdrant(client, collection_name):
    # Fetch all document IDs
    all_ids = client.scroll(collection_name=collection_name, limit=10000)[0]

    # Retrieve all documents
    all_docs = client.retrieve(collection_name=collection_name, ids=[p.id for p in all_ids])

    # Convert to LangChain Document format
    documents = []
    for doc in all_docs:
        documents.append(Document(
            page_content=doc.payload.get('chunk', ''),
            metadata={
                'file_name': doc.payload.get('file_name', 'Unknown'),  # Default to 'Unknown' if file_name is missing
                'language': doc.payload.get('language', 'Unknown')     # Default to 'Unknown' if language is missing
            }
        ))

    return documents

def generate_file_structure_from_qdrant(documents):
    """
    Generate a string representation of the file structure from Qdrant documents.
    """
    file_structure = defaultdict(set)
    for doc in documents:
        file_path = doc.metadata['file_name']
        parts = file_path.split('/')
        for i in range(len(parts)):
            parent = '/'.join(parts[:i])
            child = parts[i]
            file_structure[parent].add(child)

    def build_structure(path='', level=0):
        result = []
        indent = '    ' * level
        for item in sorted(file_structure[path]):
            full_path = f"{path}/{item}" if path else item
            result.append(f"{indent}{item}")
            if full_path in file_structure:
                result.extend(build_structure(full_path, level + 1))
        return result

    return '\n'.join(build_structure())


def query_with_langchain(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)

    # Create BM25 retriever
    all_docs = get_all_documents_from_qdrant(qdrant_client, collection_name)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 20  # Retrieve top 20 documents

    # Create vector store retriever
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=[0.5, 0.5]
    )

    # Create multi-query retriever
    llm = ChatOpenAI(model="gpt-4o")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    def retrieve_and_rerank(query):
        docs = compression_retriever.get_relevant_documents(query)

        reranked_docs = rerank_documents(docs, query, llm)

        return reranked_docs


    final_answer_template = """Based on the following information:
    Language: {language}
    Framework: {framework}
    Dependencies: {dependencies}
    File Structure:
    {file_structure}

    Original Question: {question}

    Context: {context}

    Provide a detailed answer to the question, referencing specific parts of the codebase when relevant.
    If you're unsure about any details, please state so clearly.
    """
    final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)

    context = format_docs(retrieve_and_rerank(query))

    print("Context:", context)

    final_answer_chain = final_answer_prompt | llm | StrOutputParser()

    file_structure = generate_file_structure_from_qdrant(all_docs)

    language, framework, dependencies = extract_important_project_information()

    qa_chain = (
        {
            "language": lambda _: language,
            "framework": lambda _: framework,
            "dependencies": lambda _: dependencies,
            "file_structure": lambda _: file_structure,
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | final_answer_chain
    )

    response = qa_chain.invoke(query)

    return response

def extract_important_project_information():
    all_docs = get_all_documents_from_qdrant(qdrant_client, collection_name)
    file_structure = generate_file_structure_from_qdrant(all_docs)

    llm = ChatOpenAI(model="gpt-4o")

    important_files_prompt = """
    Based on the following file structure:
    {file_structure}

    Identify the most important files to understand the project language, framework, and key components.

    Be sure to consider package managers (usually very important) and configuration files.

    Provide a list of these files in a comma-separated format. (e.g., file1, file2, file3)
    """
    important_files_chain = ChatPromptTemplate.from_template(important_files_prompt) | llm | StrOutputParser()

    retrieved_files = important_files_chain.invoke({"file_structure": file_structure})

    important_files = []

    for retrieved_file in retrieved_files.split(","):
        if retrieved_file.strip() == "":
            continue
        for doc in all_docs:
            if retrieved_file.strip() in doc.metadata['file_name']:
                important_files.append(doc)

    versions_prompt = """
    Based on the following files:
    {files}

    Generate a report of the project language version, framework version, and key components versions.

    The response format should always be in the following format, dont write anything else:
    - [Language]: [language_version]
    - [Framework]: [framework_version]

    Dependencies:
    - [dependency1]: [version1]
    - [dependency2]: [version2]
    """
    versions_chain = ChatPromptTemplate.from_template(versions_prompt) | llm | StrOutputParser()
    versions_report = versions_chain.invoke({"files": format_docs(important_files)})

    language = versions_report.split("\n")[0].split(":")[1].strip()
    framework = versions_report.split("\n")[1].split(":")[1].strip()

    dependencies = {}
    for line in versions_report.split("\n")[3:]:
        if not line.strip():
            continue
        dependency, version = line.split(":")
        dependencies[dependency.strip()] = version.strip()

    return language, framework, dependencies

def generate_dockerfile():
    language, framework, dependencies = extract_important_project_information()

    llm = ChatOpenAI(model="gpt-4o")

    dockerfile_prompt = """
    Based on the following information:
    Project Language: {language}
    Framework: {framework}
    Dependencies: {dependencies}

    Generate a Dockerfile for the project.
    """

    dockerfile_chain = ChatPromptTemplate.from_template(dockerfile_prompt) | llm | StrOutputParser()
    dockerfile = dockerfile_chain.invoke({"language": language, "framework": framework, "dependencies": dependencies})

    return dockerfile

if __name__ == "__main__":
    # Download and extract the repository
    download_and_extract_repo(owner, repo, branch, token)

    # Initialize Qdrant collection
    initialize_qdrant()

    # Process files and save their embeddings to Qdrant
    process_files_and_save_embeddings()

    # Example query
    language, framework, dependencies = extract_important_project_information()
    print(language, framework, dependencies)

    # response = query_with_langchain("How could i optimize the code?")

    # print(response)


