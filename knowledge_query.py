#credits to https://cobusgreyling.medium.com/the-langchain-implementation-of-deepminds-step-back-prompting-9d698cf3e0c2 limpeh say first
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import re
from openai import OpenAI
from neo4j import GraphDatabase, TrustCustomCAs
from flask import Flask, request, jsonify, abort
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#flask server
app = Flask(__name__)

#ssl tingz
cert_file = os.environ.get("CERT_PATH","./certs.pem")
key_file = os.environ.get("KEY_PATH","./key.pem")
ca_cert= os.environ.get("CA_PATH","./ca.pem")

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

# Setting up final prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly intelligent question answering bot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. As much as possible, base your answer on the provided context."),
    ("human", """
        Question:{question}

        Step back prompt to base your thinking on: {step_back_question}

        Context: {context}
    """)
])

final_prompt_gen = final_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
###

#NEO4J SETUP#
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), encrypted=True, trusted_certificates=TrustCustomCAs(ca_cert))

graphdb_hops = int(os.getenv("GRAPHDB_HOPS", 3))
graphdb_topk = int(os.getenv("GRAPHDB_TOPK", 50))
vectordb_topk = int(os.getenv("VECTORDB_TOPK", 5))
###

def handle_question(question):
    #step back question generation
    step_back_question = step_back_prompt(question)
    print("Step back question generated:", step_back_question)
    #retrieve and consolidate answers from multiple retrievers
    retrievers = retriever_router(question)
    print("Using retrievers:", retrievers)
    combined_context = ""
    with driver.session() as session:
        for retriever in retrievers:
            print("Using retriever:", retriever)
            if retriever == "graph_retriever":
                graph_raw_answer = graph_retriever(question,step_back_question,session)
                #process then add to dict
                cleaned_graph_answer = cleanup_graph_answer(graph_raw_answer)
                combined_context += cleaned_graph_answer["context"] + "\n"
            elif retriever == "vector_retriever":
                vector_raw_answer = vector_retriever(question,step_back_question,session)
                #process then add to dict
                cleaned_vector_answer = cleanup_vector_answer(vector_raw_answer)
                combined_context += cleaned_vector_answer["context"] + "\n"
    print("Combined context from retrievers:", combined_context)
    #final answer generation
    answer = final_prompt_gen.invoke({
        "question": question,
        "step_back_question": step_back_question,
        "context": combined_context
    })
    print("Final answer generated:", answer)
    return answer

def retriever_router(question):
    #supposed to have logic to choose retriever based on question but I lazy do today kekw
    return ["graph_retriever", "vector_retriever"]

def graph_retriever(question,step_back_question,session):
    #original question querying graphdb
    print("Querying graphdb for original question")
    original_qn_context = graphdb_interface(question,session)
    #step back question querying graphdb
    print("Querying graphdb for step back question")
    step_back_qn_context = graphdb_interface(step_back_question,session)
    return {
        "original_question_context": original_qn_context,
        "step_back_question_context": step_back_qn_context
    }

def vector_retriever(question,step_back_question,session):
    #original question querying vectordb
    embedded_original_question = embed([question])[0]
    original_qn_context = vectordb_interface(embedded_original_question,session)
    #step back question querying vectordb
    embedded_step_back_question = embed([step_back_question])[0]
    step_back_qn_context = vectordb_interface(embedded_step_back_question,session)
    return {
        "original_question_context": original_qn_context,
        "step_back_question_context": step_back_qn_context
    }


def graphdb_interface(question,session):
    print("GraphDB querying for question:", question)
    cypher = f"""
        CALL db.index.fulltext.queryNodes(
            'entityIndex',
            $query
        ) YIELD node, score
        WITH node
        MATCH p = (node)-[*1..{graphdb_hops}]-(connected)
        RETURN p LIMIT {graphdb_topk}
        """
    # print("Executing Cypher query:", cypher)
    results = session.run(cypher, {"query": question})

    subgraphs = []
    for r in results:
        # print(r)
        path = r["p"]
        segment_data = {
            "nodes": [],
            "relationships": []
        }

        for n in path.nodes:
            segment_data["nodes"].append({
                "labels": list(n.labels),
                "properties": dict(n)
            })

        for r in path.relationships:
            segment_data["relationships"].append({
                "type": r.type,
                "start_node": {
                    "labels": list(r.start_node.labels),
                    "properties": dict(r.start_node)
                },
                "end_node": {
                    "labels": list(r.end_node.labels),
                    "properties": dict(r.end_node)
                },
                "properties": dict(r)
            })
        subgraphs.append(segment_data)
    # print("Retrieved subgraphs:", subgraphs)
    return subgraphs

def vectordb_interface(embedded_question,session):
    print("VectorDB querying for embedded question.")
    cypher = f"""
        CALL db.index.vector.queryNodes(
          'doc_embeddings',
          {vectordb_topk},
          $embedding
        )
        YIELD node, score
        RETURN node.id AS id, node.text AS text, score
        ORDER BY score DESC
    """
    # print("Executing Cypher query:", cypher)
    results = session.run(cypher, {"embedding": embedded_question})

    chunks = []
    for r in results:
        chunks.append({
            "id": r["id"],
            "text": r["text"],
            "score": r["score"]
        })
    return chunks

def cleanup_graph_answer(raw_answer):
    print("Cleaning up graph answer.")
    #merge from both original and step back contexts
    merged_paths = []
    merged_paths.extend(raw_answer["original_question_context"])
    merged_paths.extend(raw_answer["step_back_question_context"])
    #deduplicate relationships
    unique_rels = set()
    facts = []

    for path in merged_paths:
        for rel in path["relationships"]:
            start = rel["start_node"]["properties"]
            end = rel["end_node"]["properties"]

            start_name = start.get("name") or start.get("title")
            end_name = end.get("name") or end.get("title")
            rel_type = rel["type"]

            if not start_name or not end_name:
                continue

            key = (start_name, rel_type, end_name)
            if key in unique_rels:
                continue

            unique_rels.add(key)

            fact = f"{start_name} {rel_type.replace('_', ' ')} {end_name}."
            facts.append(fact)
    
    graph_context = "\n".join(f"- {f}" for f in facts)

    return {
        "type": "graph",
        "facts": facts,
        "context": graph_context
    }

def cleanup_vector_answer(raw_answer):
    print("Cleaning up vector answer.")
    #merge from both original and step back contexts
    merged_chunks = []
    merged_chunks.extend(raw_answer["original_question_context"])
    merged_chunks.extend(raw_answer["step_back_question_context"])
    #remove duplicates based on id
    unique_chunks = {}
    for chunk in merged_chunks:
        cid = chunk.get("id")   # SAFE access

        if cid is None:
            # optional: log it
            # print("Skipping chunk without id:", chunk)
            continue

        if cid not in unique_chunks:
            unique_chunks[cid] = chunk
        else:
            if chunk.get("score", 0) > unique_chunks[cid].get("score", 0):
                unique_chunks[cid] = chunk

    cleaned_chunks = list(unique_chunks.values())
    #sort by score descending
    cleaned_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    #build vector answer
    context = "\n\n".join(
        f"\n-  {c['text']}"
        for c in cleaned_chunks
    )
    return {
        "type": "vector",
        "chunks": cleaned_chunks,
        "context": context
    }

def step_back_prompt(question):
    step_back_question = question_gen.invoke({"question": question})
    return step_back_question

@app.route('/client_query', methods=['POST'])
def QueryPerceptor():
    print("Received request from:", request.host)
    allowed_host = "graph-rag.han.gg"
    host = request.host.split(":")[0]
    print(host)
    if host != allowed_host:
        abort(403, description="Forbidden: Invalid Host header")
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field in JSON request"}), 400
        else:
            print(data)
            print("Processing question:", data['question'])
            question = data['question']
            
            response = handle_question(question)

            return jsonify({
                "response": response
            })
    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5112, ssl_context=(cert_file, key_file))