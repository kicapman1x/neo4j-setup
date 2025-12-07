#!/bin/bash

#using python virtual env
source $PYTHON_VENV_DIR/bin/activate

#install requirements 
pip install -r requirements.txt

python3 knowledge_ingest.py