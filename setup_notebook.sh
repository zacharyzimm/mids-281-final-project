#!/bin/bash


# create venv
python -m venv venv
source venv/bin/activate


#install dependencies from requirements.txt
pip install -r requirements.txt

#make sure notebook can use the venv
python -m ipykernel install --name=venv

#start jupyter notebook
jupyter notebook
