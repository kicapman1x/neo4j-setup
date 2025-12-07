import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import re
from openai import OpenAI

load_dotenv()
raw_dir="/home/daddy/apps/neo4j/knowledge/raw"
proc_dir="/home/daddy/apps/neo4j/knowledge/processed"

open_ai_client = OpenAI()

def process_files(raw_dir):
    print("Processing files from:", raw_dir)
    list_of_files = os.listdir(raw_dir)
    for file_name in list_of_files:
        print("Processing file:", file_name)
        file_path = os.path.join(raw_dir, file_name)
        with open(file_path, "r") as f:
            doc_text = f.read()
            embeddings = embed(chunk_text(doc_text))
            with open(os.path.join(proc_dir, "embeddings"), "a") as embd_f:
                for embd in embeddings:
                    embd_f.write(json.dumps(embd) + "\n")
            with open(os.path.join(proc_dir, file_name), "w") as out_f:
                out_f.write(process_data(doc_text))
                print("Processed data written to:", os.path.join(proc_dir, file_name))
    print ("WE OUT!")


def process_data(doc_text):
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0
    )
    schema = {
        "nodes": [
            {"type": "Ministry", "name": ""},
            {"type": "Document", "title": "", "date": "", "source": "", "type": ""},
            {"type": "Person", "name": ""},
            {"type": "Program", "name": ""},
            {"type": "Location", "name": ""},
            {"type": "Penalty", "law": "", "fine": "", "imprisonment": ""}
        ],
        "relationships": [
            {"from": "", "to": "", "type": ""}
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

if __name__ == "__main__":
    process_files(raw_dir)