# mids-281-final-project
Shared Repo for Final Project


This is the repo for our final project! To get the files locally on your computer and get started, please do the following:

NOTE: these instructions are for Mac and Linux users

1. Install git-lfs on your computer. This will enable you to download the .zip file containing the images (it's only 119 MB, should have no problem with a local download)
    1. go to https://git-lfs.com/ and follow the instructions. If you're on Mac and have homebrew installed, just run `brew install git-lfs`
2. in your MIDS 281 folder (or folder of your choice) clone the repo
    1. `git clone https://github.com/zacharyzimm/mids-281-final-project.git`
3. go into the directory and pull the zip file (don't worry about unzipping it, the setup script will do that for you)
    1. `cd mids-281-final-project`
    2. `git lfs pull`
4. make sure the setup script is executable
    1. `chmod +x setup_notebook.sh`
5. run the setup script. It will prompt you to enter your password since it needs permission to download the unzip package if you don't have it
    1. `./setup_notebook.sh`
  
The setup script will install `unzip`, unzip the image files to a folder called `Data`, setup a venv with opencv installed, and open a jupyter notebook
To get started working, select New -> venv in the jupyter notebook interface which will open a notebook that uses said venv. Happy coding!
