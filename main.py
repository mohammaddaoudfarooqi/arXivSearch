"""
This script provides a comprehensive solution for searching and summarizing academic papers from arXiv using a combination of full-text and vector-based search techniques. It leverages MongoDB for storage, AWS Bedrock for embeddings, and Gradio for a user interface.
Modules and Functions:
- `setup_vector_search_index()`: Sets up the vector search index and search index for the specified MongoDB collection.
- `hybrid_search(query, vector_query, weight=0.5, top_n=10)`: Performs a hybrid search with semantic boosting on MongoDB.
- `fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]`: Fetches papers from arXiv based on a search query.
- `download_papers(state: Dict) -> Dict`: Downloads papers from arXiv, processes them, and stores them in MongoDB.
- `perform_hybrid_search(state: Dict) -> Dict`: Performs a hybrid search using both full-text and vector-based search.
- `find_search_topic(state: str) -> Dict`: Identifies the core search topic from a user's query for academic papers on arXiv.
- `generate_answer(state: str) -> Dict`: Generates an answer to a user's query based on retrieved excerpts from arXiv papers.
- `check_documents_extracted(state: Dict) -> str`: Determines the next action based on the state of document extraction.
- `check_downloaded_papers(state: Dict) -> str`: Checks the state of downloaded papers and returns a status string.
- `create_graph_nodes()`: Creates and configures the nodes and edges for the workflow graph.
- `handle_query(question: str, history: str)`: Handles a query by streaming the output from the app and yielding the results.
Classes:
- `GraphState(TypedDict)`: Represents the state of a graph search operation.
Gradio Interface:
- Defines a Gradio interface for interacting with the ArXiv Research Assistant.
Execution:
- The script sets up the vector search index, creates graph nodes, compiles the workflow, and launches the Gradio app.

Author:
    Mohammad Daoud Farooqi
"""

import json
import os
import time
import urllib.parse
from typing import Dict, List

import arxiv
import boto3
import gradio as gr
import pymongo
from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v1"
COMPLETION_MODEL_NAME = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = "mongodb_arxiv"
MONGODB_COLLECTION = "documents"
VECTOR_SEARCH_INDEX_NAME = "vector_index"
SEARCH_INDEX_NAME = "full_text_index"

# Initialize MongoDB client
collection = MongoClient(MONGODB_URI)[MONGODB_DB_NAME][MONGODB_COLLECTION]

# Setup AWS and Bedrock client
config = Config(read_timeout=100000)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=config)

# Initialize embedding model
embedding_model = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_NAME, client=bedrock_client
)


# Setup vector search index
def setup_vector_search_index():
    """
    Sets up the vector search index and search index for the specified collection.
    This function checks if the collection exists in the database. If not, it creates the collection.
    It then defines two index configurations:
    1. A vector search index for the "embeddings" field with cosine similarity.
    2. A search index for the "text" field with string type.
    The function attempts to create these indexes in the collection. If an error occurs during the creation
    of an index, it is silently ignored.
    Raises:
        pymongo.errors.PyMongoError: If there is an error creating the search index.
    """
    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)

    index_definitions = [
        {
            "name": VECTOR_SEARCH_INDEX_NAME,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embeddings",
                        "numDimensions": 1536,
                        "similarity": "cosine",
                    }
                ]
            },
        },
        {
            "name": SEARCH_INDEX_NAME,
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {"text": {"type": "string"}},
                }
            },
        },
    ]

    for index_definition in index_definitions:
        try:
            collection.create_search_index(model=index_definition)
            time.sleep(10)
        except pymongo.errors.PyMongoError:
            pass


def hybrid_search(query, vector_query, weight=0.5, top_n=10):
    """
    Perform hybrid search with semantic boosting on MongoDB.

    :param query: The full-text search query
    :param vector_query: The vector search query
    :param weight: Weight for semantic (vector) score (0-1)
    :param top_n: Limit for top N results in vector search
    :return: Aggregation pipeline results
    """
    # Ensure weight is between 0 and 1
    assert 0 <= weight <= 1, "Weight must be between 0 and 1"
    pipeline = [
        {
            "$search": {
                "index": "full_text_index",
                "text": {"query": query, "path": "text"},
            }
        },
        {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
        {"$setWindowFields": {"output": {"maxScore": {"$max": "$fts_score"}}}},
        {
            "$addFields": {
                "normalized_fts_score": {"$divide": ["$fts_score", "$maxScore"]}
            }
        },
        {"$project": {"text": 1, "normalized_fts_score": 1}},
        {
            "$unionWith": {
                "coll": collection.name,
                "pipeline": [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_SEARCH_INDEX_NAME,
                            "queryVector": vector_query,  # Parameterized vector query
                            "path": "embeddings",
                            "numCandidates": 200,
                            "limit": top_n,  # Limit for vector search results
                        }
                    },
                    {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
                    {
                        "$setWindowFields": {
                            "output": {"maxScore": {"$max": "$vs_score"}}
                        }
                    },
                    {
                        "$addFields": {
                            "normalized_vs_score": {
                                "$divide": ["$vs_score", "$maxScore"]
                            }
                        }
                    },
                    {"$project": {"text": 1, "normalized_vs_score": 1}},
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",  # Replace with the unique identifier field
                "fts_score": {"$max": "$normalized_fts_score"},
                "vs_score": {"$max": "$normalized_vs_score"},
                "text_field": {"$first": "$text"},
            }
        },
        {
            "$addFields": {
                "hybrid_score": {
                    "$add": [
                        {"$multiply": [weight, {"$ifNull": ["$vs_score", 0]}]},
                        {"$multiply": [1 - weight, {"$ifNull": ["$fts_score", 0]}]},
                    ]
                }
            }
        },
        {"$sort": {"hybrid_score": -1}},
        {
            "$limit": top_n  # Limit for final hybrid results
        },
        {
            "$project": {
                "_id": 0,  # Exclude the ID field from the output
                "fts_score": 1,
                "vs_score": 1,
                "score": "$hybrid_score",
                "text": "$text_field",
            }
        },
    ]

    # Execute the pipeline
    results = list(collection.aggregate(pipeline))
    return results


# Fetch papers from ArXiv
def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Fetches papers from arXiv based on a search query.

    Args:
        query (str): The search query string.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        List[Dict]: A list of dictionaries, each containing details of a paper:
            - title (str): The title of the paper.
            - authors (List[str]): A list of author names.
            - summary (str): The summary of the paper.
            - published (str): The publication date in ISO format.
            - pdf_url (str): The URL to the PDF of the paper.
            - arxiv_url (str): The URL to the paper's arXiv entry.
    """
    print("fetch_arxiv_papers", query)
    query = urllib.parse.quote(f"all:{query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    # Use the Client for searching
    client = arxiv.Client()

    # Execute the search
    searchResults = client.results(search)

    return [
        {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "published": paper.published.isoformat(),
            "pdf_url": paper.pdf_url,
            "arxiv_url": paper.entry_id,
        }
        for paper in searchResults
    ]


def download_papers(state: Dict) -> Dict:
    """
    Downloads papers from arXiv based on the given state, processes them, and stores them in MongoDB.

    Args:
        state (Dict): A dictionary containing the state information, including the topic to search for papers.

    Returns:
        Dict: The result of performing a hybrid search on the updated state.

    The function performs the following steps:
    1. Fetches papers from arXiv based on the topic specified in the state.
    2. Processes each paper by converting it to a string and generating embeddings using an embedding model.
    3. Stores each processed paper in a MongoDB collection, updating existing entries or inserting new ones.
    4. Waits for 10 seconds to avoid overwhelming the server.
    5. Performs a hybrid search on the updated state and returns the result.
    """
    papers = fetch_arxiv_papers(state["topic"])
    # Store papers in MongoDB
    for paper in papers:
        paper["text"] = str(paper)
        paper["embeddings"] = embedding_model.embed_query(str(paper))
        collection.update_one({"title": paper["title"]}, {"$set": paper}, upsert=True)
    time.sleep(10)
    return perform_hybrid_search(state)


# Perform vector search
def perform_hybrid_search(state: Dict) -> Dict:
    """
    Perform a hybrid search using both full-text search and vector-based search.

    Args:
        state (Dict): A dictionary containing the search state, including the topic to search for.

    Returns:
        Dict: A dictionary containing the search results. If no relevant documents are found,
              the dictionary will contain a message indicating no documents were found. Otherwise,
              it will contain a list of relevant document texts.
    """
    vector_query = embedding_model.embed_query(state["topic"])
    query = state["topic"]
    documents = hybrid_search(query, vector_query, weight=0.8, top_n=5)

    # Drop documents with cosine similarity score < 0.8
    # Documents with a cosine similarity score below 0.8 are considered less relevant
    # and are therefore excluded from the results to ensure higher quality matches.
    relevant_results = [doc["text"] for doc in documents if doc["score"] >= 0.95]
    if not relevant_results:
        return {"documents": "No documents found"}
    else:
        return {"documents": relevant_results}


# Generate answer
def find_search_topic(state: str) -> Dict:
    """
    Identifies the core search topic from a user's query for academic papers on arXiv.

    This function processes a user's question or statement to extract the main topic or keywords
    relevant to searching academic papers on arXiv. It follows specific guidelines to focus on
    the core topic, ignore irrelevant details, and handle multiple topics appropriately.

    Args:
        state (str): A dictionary containing the user's query under the key 'question'.

    Returns:
        Dict: A dictionary with the extracted topic or keywords under the key 'topic'.
    """

    context = f"""
        You are a smart assistant tasked with identifying the core search topic from user queries. Given a user's question or statement, extract the main topic or keywords relevant to searching academic papers on arXiv. 

        **Guidelines:**
        1. Focus on the **core topic** or **keywords** that capture the essence of the query.
        2. Ignore any irrelevant details or context unrelated to the search.
        3. If the query mentions multiple topics, prioritize the most central one or list them as separate topics.
        4. Be concise and provide only the topic(s) or keyword(s).

        **Examples**:

        1. **Input**: "What are the latest advancements in quantum machine learning algorithms?"  
        **Output**: "Quantum machine learning algorithms"

        2. **Input**: "Can you find papers about using reinforcement learning in robotics?"  
        **Output**: "Reinforcement learning in robotics"

        3. **Input**: "I'm interested in understanding how transformers improve NLP tasks."  
        **Output**: "Transformers in NLP"

        4. **Input**: "What are the challenges in applying GANs to image synthesis?"  
        **Output**: "GANs for image synthesis"

        5. **Input**: "Summarize research on graph neural networks for drug discovery."  
        **Output**: "Graph neural networks for drug discovery"

        6. **Input**: "Explain the impact of self-supervised learning in computer vision."  
        **Output**: "Self-supervised learning in computer vision"

        ---

        **Now process the following query and extract the search topic:**

        User Query: {state['question']} 
        Output: Provide response as the core topic or keywords relevant to searching academic papers on arXiv. The output should only contain the topic and nothing else.

        """

    # Message payload
    messages = [{"role": "user", "content": f"{context}"}]
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20000,
        "temperature": 0.7,
        "messages": messages,
    }
    response = bedrock_client.invoke_model(
        modelId=COMPLETION_MODEL_NAME, body=json.dumps(request_payload)
    )
    response_text = json.loads(response["body"].read())["content"][0]["text"]
    print({"topic": response_text})
    return {"topic": response_text}


def generate_answer(state: str) -> Dict:
    """
    Generates an answer to a user's query based on retrieved excerpts from arXiv papers.
    Args:
        state (str): A dictionary containing the user's question and the retrieved documents.
            - 'question': The user's query or topic of interest.
            - 'documents': A list of excerpts or relevant sections from arXiv papers.
    Returns:
        Dict: A dictionary containing the generated answer.
            - 'answer': The response to the user's query based on the retrieved papers.
    The function constructs a context for the model, including the user's query and the retrieved papers,
    and sends a request to the model to generate a response. The response is then parsed and returned.
    Example:
        state = {
            'question': "What are the primary challenges in applying reinforcement learning to autonomous vehicles?",
            'documents': [
                "Excerpt from 'Deep Reinforcement Learning for Autonomous Driving': '...the main challenge is the trade-off between exploration and safety in high-stakes environments...'",
                "Excerpt from 'Safe Reinforcement Learning Methods': '...limited training data in realistic driving scenarios poses significant hurdles...'"
            ]
        result = generate_answer(state)
        print(result['answer'])
    """

    context = (
        """
    You are an expert assistant trained on scientific literature, specialized in helping users understand and summarize research papers from arXiv. Your role is to:

    1. **Retrieve Relevant Information**:
    - Leverage provided excerpts from arXiv papers to answer user queries accurately and concisely.
    - Focus on clarity and relevance, even for complex or technical topics.

    2. **Synthesize Responses**:
    - Combine information from multiple sources when needed to provide a comprehensive answer.
    - If requested, offer detailed explanations, simplified summaries, or comparisons across research findings.

    3. **Maintain Scientific Rigor**:
    - Ensure accuracy by relying only on the retrieved content and explicitly indicate any uncertainties or assumptions.
    - Reference key papers or sections when applicable.

    **Example Tasks**:
    1. **Summarization**: "Summarize the key findings of recent papers on graph neural networks for drug discovery."
    2. **Comparison**: "How does the transformer architecture compare to RNNs for language modeling, according to the retrieved papers?"
    3. **Explanation**: "Explain the main contribution of the paper titled 'Attention is All You Need' in simple terms."
    4. **Trend Analysis**: "What are the emerging trends in quantum machine learning research?"
    5. **Code Assistance**: "What does the pseudocode in the retrieved paper on GANs signify, and how can it be implemented?"

    **Input Format**:
    - User Query: [User's question or topic of interest]
    - Retrieved Papers: [List of excerpts or relevant sections from arXiv papers]

    **Output Requirements**:
    1. **Clear and Structured Response**:
    - Provide answers in a structured format with sections like "Summary," "Key Insights," or "Step-by-Step Explanation."
    2. **Citations**: Include references to the titles or sections of the papers cited in your response. Also include the pdf link.
    3. **Contextual Relevance**: Address the user's specific needs and tailor the response to their level of expertise.

    **Constraints**:
    - Do not fabricate information or cite papers that are not retrieved.
    - If the query cannot be answered with the provided information, clearly state the limitation and suggest potential follow-ups.

    ---

    **Example Input**:  
    User Query: "What are the primary challenges in applying reinforcement learning to autonomous vehicles?"  
    Retrieved Papers:  
    1. Excerpt from "Deep Reinforcement Learning for Autonomous Driving": "...the main challenge is the trade-off between exploration and safety in high-stakes environments..."  
    2. Excerpt from "Safe Reinforcement Learning Methods": "...limited training data in realistic driving scenarios poses significant hurdles..."  

    **Example Output**:  
    **Summary**:  
    - Reinforcement learning (RL) in autonomous vehicles faces challenges such as ensuring safety during exploration and managing limited training data for realistic scenarios.  
    **Key Insights**:  
    1. Safety vs. Exploration: Balancing learning efficiency with safety concerns in real-world environments.  
    2. Data Scarcity: A lack of diverse and realistic driving data hampers robust training and validation.  
    **References**:  
    - "Deep Reinforcement Learning for Autonomous Driving"  
    - "Safe Reinforcement Learning Methods"

    ---
        
    """
        + f"- User Query: {state['question']}\n"
        + f"- Retrieved Papers: {state['documents']}\n"
        + "Output: Provide response to the user query based on the retrieved papers."
    )

    # Message payload
    messages = [{"role": "user", "content": f"{context}"}]
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000000,
        "temperature": 0.7,
        "messages": messages,
    }
    response = bedrock_client.invoke_model(
        modelId=COMPLETION_MODEL_NAME, body=json.dumps(request_payload)
    )
    response_text = json.loads(response["body"].read())["content"][0]["text"]
    return {"answer": response_text}


# Check documents
def check_documents_extracted(state: Dict) -> str:
    """
    Determines the next action based on the state of document extraction.

    Args:
        state (Dict): A dictionary containing the state of document extraction.

    Returns:
        str: "download_paper" if no documents are found, otherwise "generate_answer".
    """
    return (
        "download_paper"
        if state["documents"] == "No documents found"
        else "generate_answer"
    )


def check_downloaded_papers(state: Dict) -> str:
    """
    Checks the state of downloaded papers and returns a status string.

    Args:
        state (Dict): A dictionary containing the state information,
                      specifically with a key "documents".

    Returns:
        str: Returns "no_paper_found" if no documents are found,
             otherwise returns "hybrid_search".
    """
    return (
        "no_paper_found"
        if state["documents"] == "No documents found"
        else "hybrid_search"
    )


# Define graph workflow
class GraphState(TypedDict):
    """
    GraphState is a TypedDict that represents the state of a graph search operation.

    Attributes:
        question (str): The question being asked in the graph search.
        topic (str): The topic related to the question.
        documents (List[Dict]): A list of documents related to the topic and question.
        answer (str): The answer derived from the graph search.
    """

    question: str
    topic: str
    documents: List[Dict]
    answer: str


workflow = StateGraph(GraphState)


def create_graph_nodes():
    """
    Creates and configures the nodes and edges for the workflow graph.

    This function sets up the workflow by adding nodes and defining the
    relationships between them through edges and conditional edges. The
    workflow consists of the following nodes:
    - "find_search_topic": Identifies the search topic.
    - "hybrid_search": Performs a hybrid search.
    - "download_paper": Downloads the papers.
    - "generate_answer": Generates the answer.

    The edges and conditional edges define the flow of the workflow:
    - START -> "find_search_topic"
    - "find_search_topic" -> "hybrid_search"
    - Conditional edges from "hybrid_search" based on `check_documents_extracted`:
        - "download_paper" if documents are extracted.
        - "generate_answer" if no documents are extracted.
    - Conditional edges from "download_paper" based on `check_downloaded_papers`:
        - "hybrid_search" if papers are downloaded.
        - "no_paper_found" (END) if no papers are found.
    - "generate_answer" -> END
    """
    workflow.add_node("find_search_topic", find_search_topic)
    workflow.add_node("hybrid_search", perform_hybrid_search)
    workflow.add_node("download_paper", download_papers)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_edge(START, "find_search_topic")
    workflow.add_edge("find_search_topic", "hybrid_search")
    workflow.add_conditional_edges(
        "hybrid_search",
        check_documents_extracted,
        {"download_paper": "download_paper", "generate_answer": "generate_answer"},
    )
    workflow.add_conditional_edges(
        "download_paper",
        check_downloaded_papers,
        {"hybrid_search": "hybrid_search", "no_paper_found": END},
    )
    workflow.add_edge("generate_answer", END)


setup_vector_search_index()
create_graph_nodes()
app = workflow.compile()


def handle_query(question: str, history: str):
    """
    Handles a query by streaming the output from the app and yielding the results.

    Args:
        question (str): The question to be processed.
        history (str): The history of previous interactions.

    Yields:
        str: The streamed output data from the app, including node information and answers.
    """
    try:
        inputs = {"question": question}
        config = {"configurable": {"thread_id": "1"}}
        output_data = []
        for output in app.stream(inputs, config):
            for key, value in output.items():
                print(f"Node {key}:")
                print(value)
                output_data.append(f"Node {key}:\n{value}\n")
                yield "\n".join(output_data)
        try:
            yield str(value["answer"])
        except Exception:
            pass
    except Exception as e:
        yield f"An error occurred: {str(e)}"


# Define the Gradio interface
with gr.Blocks(
    fill_height=True,
    fill_width=True,
    title="ArXiv Research Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.ChatInterface(
        fn=handle_query,
        type="messages",
        title="ArXiv Research Assistant",
        description="<center>Get assistance in searching and summarizing academic papers from arXiv.</center>",
        multimodal=False,
        fill_height=True,
        fill_width=False,
        show_progress=False,
        concurrency_limit=None,
        examples=[
            "Tell me about Hallucinations in Large Language Models.",
            "What are the latest advancements in quantum computing?",
            "Explain the role of reinforcement learning in autonomous driving systems.",
            "Summarize recent research on graph neural networks for social network analysis.",
            "How is self-supervised learning transforming computer vision tasks?",
            "What are the primary applications of GANs in medical imaging?",
            "Tell me about 'Attention is All You Need' by Vaswani et al. and its impact on NLP.",
            "Summarize the key findings of 'Retrieval-Augmented Generation for Large Language Models: A Survey'.",
            "Explain the contributions of the paper titled 'RL-STaR: Theoretical Analysis of Reinforcement Learning Frameworks for Self-Taught Reasoner'.",
            "What are the core challenges highlighted in 'Deep Reinforcement Learning for Autonomous Driving'?",
            "Compare the effectiveness of RNNs and Transformers in language modeling.",
            "What trends are emerging in AI research for natural language processing?",
            "How does diffusion modeling differ from traditional GANs for image generation?",
            "Describe the methods used in zero-shot learning for NLP tasks.",
            "What algorithms are commonly employed for multi-agent reinforcement learning?",
            "Explain the theory behind policy gradient methods in reinforcement learning.",
            "How do quantum annealing techniques contribute to optimization problems?",
            "What improvements do pre-trained language models like GPT bring to machine translation?",
            "Explain how retrieval-augmented generation works in large language models.",
            "I’m curious about the use of transformers in computer vision. Can you summarize the latest research?",
            "What does the literature say about ethical considerations in AI development?",
        ],
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7854,
        ssl_verify=False,
        share=False,
        width="100%",
        height="100%",
    )
