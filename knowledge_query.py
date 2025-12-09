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
            step_back_question = question_gen.invoke({"question": question})

            return jsonify({
                "original_question": question,
                "step_back_question": step_back_question
            })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5112, ssl_context=(cert_file, key_file))