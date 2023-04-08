# Environment
This documents go through the process of managing environments. 

## Create a python virtual environment

### From zero
1. Go to the project directory
2. Create the environment 
    > python -m venv venv
3. Activate the environment
    > .\venv\Scripts\activate

### Install libraries existing libraries into the environment
pip install -r requirements.txt

## (re)create requirements.txt
### From our local python virtual environment
> pip list --format=freeze > requirements.txt

## Links
### requirements.txt vs setup
https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af#:~:text=The%20requirements.&text=txt%20is%20a%20file%20listing,be%20pinned%20or%20non%2Dpinned.

### create requirements.txt from conda environment
https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3
