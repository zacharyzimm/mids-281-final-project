#!/bin/bash

# install unzip and unzip the large file
sudo apt-get unzip
unzip chest-ctscan-images.zip

# create venv
python3 -m venv venv
source venv/bin/activate


#install dependencies from requirements.txt
pip install -r requirements.txt


#make sure notebook can use the venv
python3 -m ipykernel install --name=venv

#start jupyter notebook
jupyter notebook
