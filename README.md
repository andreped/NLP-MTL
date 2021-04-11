# NLP-MTL

Simple project of which I trained an NLP for classification of free text using multi-task learning, effectively being able to solve multiple tasks at once in the same network.

The project was part of the Cogito NLP workshop Spring 2020.

Simply make virtual environment, activate it, and install dependencies:
```
virtualenv -ppython3 venv --clear
sourve venv/bin/activate
pip install -r requirements.txt
```

Then go inside the workshop folder, and simply run the train script to preprocess data, and train and evaluate trained model:
```
cd cogito-nlp-workshop-spring-2020/
python train.py
```
