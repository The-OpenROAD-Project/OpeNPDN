#!/bin/bash

python3 -m venv PDN-opt 
echo "Setting up virtual environment"
source PDN-opt/bin/activate
which python3
echo "Installing all the necessary pacakges"
pip3 install -r requirements.txt --no-cache-dir
source PDN-opt/bin/activate
