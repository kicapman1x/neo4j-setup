#credits to https://cobusgreyling.medium.com/the-langchain-implementation-of-deepminds-step-back-prompting-9d698cf3e0c2 limpeh say first
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import re
from openai import OpenAI
from neo4j import GraphDatabase
from flask import Flask, request, jsonify, abort
from langchain_core.prompts.chat import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#flask server
app = Flask(__name__)
logger = logging.getLogger(__name__)

#ssl tingz
cert_file = os.environ.get("CERT_PATH","./certs.pem")
key_file = os.environ.get("KEY_PATH","./key.pem")
ca_file= os.environ.get("CA_PATH","./ca.pem")

#OPENAI SETUP#
# Setting up embedding model
open_ai_client = OpenAI()

def embed(texts):
    response = open_ai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
        )
    return list(map(lambda n: n.embedding, response.data))

# Setting up few shot prompt message
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?"
    },
    {
        "input": "Jan Sindel’s was born in what country?", 
        "output": "what is Jan Sindel’s personal history?"
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
    few_shot_prompt,
    ("user", "{question}"),
])

question_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
###

#NEO4J SETUP#
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), encrypted=True, trusted_certificates=TrustCustomCAs(ca_cert))

graphdb_hops = os.getenv("GRAPHDB_HOPS", 3)
graphdb_topk = os.getenv("GRAPHDB_TOPK", 50)
vectordb_topk = os.getenv("VECTORDB_TOPK", 5)
###

def handle_question(question):
    #retrieve and consolidate answers from multiple retrievers
    retrievers = retriever_router(question)
    with driver.session() as session:
        for retriever in retrievers:
            if retriever == "graph_retriever":
                graph_raw_answer = graph_retriever(question,session)
                #process then add to dict
            elif retriever == "vector_retriever":
                vector_raw_answer = vector_retriever(question,session)
                #process then add to dict
        #combine answers? 

def retriever_router(question):
    #supposed to have logic to choose retriever based on question but I lazy do today kekw
    return ["graph_retriever", "vector_retriever"]

def graph_retriever(question,session):
    #original question querying graphdb
    original_qn_context = graphdb_interface(question,session)
    #step back question querying graphdb
    step_back_question = step_back_prompt(question)
    step_back_qn_context = graphdb_interface(step_back_question,session)
    return {
        "original_question_context": original_qn_context,
        "step_back_question_context": step_back_qn_context
    }

def vector_retriever(question,session):
    #original question querying vectordb
    embedded_original_question = embed([question])[0]
    original_qn_context = vectordb_interface(embedded_original_question,session)
    #step back question querying vectordb
    step_back_question = step_back_prompt(question)
    embedded_step_back_question = embed([step_back_question])[0]
    step_back_qn_context = vectordb_interface(embedded_step_back_question,session)
    return {
        "original_question_context": original_qn_context,
        "step_back_question_context": step_back_qn_context
    }


def graphdb_interface(question,session):
    cypher = f"""
        CALL db.index.fulltext.queryNodes(
            'entityIndex',
            {question}
        ) YIELD node, score
        WITH node
        MATCH p = (node)-[*1..{graphdb_hops}]-(connected)
        RETURN p LIMIT {graphdb_topk}
        """
    result = session.run(cypher)

    subgraphs = []
    for record in results:
        path = record["p"]
        segment_data = {
            "nodes": [],
            "relationships": []
        }

        for n in path.nodes:
            segment_data["nodes"].append(dict(n))

        for r in path.relationships:
            start_node = path.nodes[r.start_node.id]
            end_node = path.nodes[r.end_node.id]
            segment_data["relationships"].append({
                "type": r.type,
                "start_node": dict(start_node),
                "end_node": dict(end_node),
                "properties": dict(r)
            })

        subgraphs.append(segment_data)
    return subgraphs

def vectordb_interface(embedded_question,session):
    cypher = f"""
        CALL db.index.vector.queryNodes(
          'doc_embeddings',
          {vectordb_topk},
          {embedded_question}
        )
        YIELD node, score
        RETURN node.id AS id, node.text AS text, score
        ORDER BY score DESC
    """
    result = session.run(cypher, embedding=embedded_question)

    chunks = []
    for r in results:
        chunks.append({
            "id": r["id"],
            "text": r["text"],
            "score": r["score"]
        })
    return chunks

def step_back_prompt(question):
    step_back_question = question_gen.invoke({"question": question})
    return step_back_question

@app.route('/client_query', methods=['POST'])
def QueryPerceptor():
    allowed_host = "graph-rag.han.gg"
    host = request.host.split(":")[0]
    if host != allowed_host:
        logger.warning(f"Unauthorized request to {request.host}")
        abort(403, description="Forbidden: Invalid Host header")
    try:
        data = request.get_json()
        logger.debug(f'Data received: {data}')
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field in JSON request"}), 400
        else:
            question = data['question']
            
            response = handle_question(question)

            return jsonify({
                "original_question": question,
                "step_back_question": step_back_question
            })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5112, ssl_context=(cert_file, key_file))