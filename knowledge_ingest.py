import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import re
from openai import OpenAI
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv()
raw_dir = os.getenv("READ_PATH")
proc_dir = os.getenv("WRITE_PATH")

open_ai_client = OpenAI()

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

ca_cert = os.getenv("CA_PATH")

def process_files(raw_dir):
    print("Processing files from:", raw_dir)
    list_of_files = os.listdir(raw_dir)
    for file_name in list_of_files:
        print("Processing file:", file_name)
        file_path = os.path.join(raw_dir, file_name)
        with open(file_path, "r") as f:
            doc_text = f.read()
            # chunks = chunk_text(doc_text)
            # embeddings = embed(chunks)
            # with open(os.path.join(proc_dir, "chunks"), "a") as chnk_f:
            #     for chunk in chunks:
            #         chnk_f.write(json.dumps(chunk) + "\n")
            # with open(os.path.join(proc_dir, "embeddings"), "a") as embd_f:
            #     for embd in embeddings:
            #         embd_f.write(json.dumps(embd) + "\n")
            with open(os.path.join(proc_dir + "/nodes", file_name), "w") as out_f:
                out_f.write(process_data(doc_text))
                print("Processed data written to:", os.path.join(proc_dir + "/nodes", file_name))
    print ("WE OUT!")


def process_data(doc_text):
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0
    )
    schema = {
        "nodes": [
            {"node_type": "Ministry", "name": ""},
            {"node_type": "Document", "title": "", "date": "", "source": "", "doc_type": ""},
            {"node_type": "Person", "name": ""},
            {"node_type": "Program", "name": ""},
            {"node_type": "Location", "name": ""},
            {"node_type": "Penalty", "law": "", "fine": "", "imprisonment": ""}
        ],
        "relationships": [
            {"from": "", "to": "", "rel_type": ""}
        ]
    }

    prompt_template = """
    Extract nodes and relationships from the document in JSON.
    Follow this schema exactly:
    {schema}

    Document:
    {document_text}

    Return valid JSON only.
    """

    prompt = PromptTemplate(
        input_variables=["doc_text", "schema"],
        template=prompt_template
    )

    formatted_prompt = prompt.format(document_text=doc_text, schema=json.dumps(schema, indent=2))
    message = HumanMessage(content=formatted_prompt)
    response = llm.generate([[message]])
    print(response)
    raw_output = response.generations[0][0].text

    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Failed to parse LLM output as JSON:")
        print(raw_output)
        parsed_output = None

    return json.dumps(parsed_output, indent=2)

def chunk_text(text, chunk_size=500, overlap=40):
    cleaned_text = re.sub(r'[^\S\n]+', ' ', text)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(cleaned_text[start:].strip())
            break

        while end < text_length and cleaned_text[end] != " ":
            end += 1

        chunk = cleaned_text[start:end].strip()
        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0
    return chunks

def embed(texts):
    response = open_ai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
        )
    return list(map(lambda n: n.embedding, response.data))

def ingest_to_neo4j(proc_dir):
    print("Ingesting data to Neo4j from:", proc_dir)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), encrypted=True, trusted_certificates=TrustCustomCAs(ca_cert))
    with driver.session() as session:
        #vectordb
        # ingest_for_vectordb(session, proc_dir)
        #graphdb
        ingest_for_graphdb(session, proc_dir)
    driver.close()
    print("Data ingestion to Neo4j completed.")

def ingest_for_vectordb(session, proc_dir):
    chunks_file = os.path.join(proc_dir, "chunks")
    embeddings_file = os.path.join(proc_dir, "embeddings")

    with open(chunks_file, "r") as cf, open(embeddings_file, "r") as ef:
        for i, (chunk, emb) in enumerate(zip(cf, ef)):
            chunk = chunk.strip()
            emb = json.loads(emb)
            session.run(
                """
                MERGE (d:Doc {id: $id})
                SET d.text = $text,
                    d.embedding = $embedding
                """,
                id=i,
                text=chunk,
                embedding=emb
            )

def ingest_for_graphdb(session, proc_dir):
    nodes_dir = os.path.join(proc_dir, "nodes")
    for f in os.listdir(nodes_dir):
        file_path = os.path.join(nodes_dir, f)
        print("Ingesting nodes and relationships from file:", file_path)
        data = json.load(open(file_path, "r"))
        ingest_file(session, data)

def ingest_file(session, data):
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])

    #node data
    for node in nodes:
        node_type = node.get("node_type")
        if not node_type:
            continue

        if node_type in ["Ministry", "Person", "Program", "Location"]:
            merge_field = "name"
            merge_value = node.get("name")

        elif node_type == "Document":
            merge_field = "title"
            merge_value = node.get("title")

        elif node_type == "Penalty":
            merge_field = "law"
            merge_value = node.get("law")

        else:
            merge_field = "name"
            merge_value = node.get("name")

        if not merge_value:
            print(f"Skipping node with missing merge key: {node}")
            continue

        props = {
            k: v for k, v in node.items()
            if k not in ["node_type", merge_field] and v not in ["", None]
        }

        cypher_node = f"""
        MERGE (n:{node_type} {{ {merge_field}: $merge_value }})
        SET n += $props
        """
        session.run(cypher_node, merge_value=merge_value, props=props)

    #relationship data
    for rel in relationships:
        from_name = rel.get("from")
        to_name = rel.get("to")
        rel_type = rel.get("rel_type")

        cypher_rs = f"""
        MATCH (a)
        WHERE 
            (a.name IS NOT NULL AND a.name = $from_name) OR
            (a.title IS NOT NULL AND a.title = $from_name) OR
            (a.law IS NOT NULL AND a.law = $from_name)

        MATCH (b)
        WHERE 
            (b.name IS NOT NULL AND b.name = $to_name) OR
            (b.title IS NOT NULL AND b.title = $to_name) OR
            (b.law IS NOT NULL AND b.law = $to_name)

        MERGE (a)-[r:`{rel_type}`]->(b)
        """
        session.run(cypher_rs, from_name=from_name, to_name=to_name)

if __name__ == "__main__":
    process_files(raw_dir)
    ingest_to_neo4j(proc_dir)