# NLP-MTL
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

Code from a workshop of which I trained an NLP to solve multiple classification tasks simultaneously from free text. For this I trained a neural network using multi-task learning (MTL), using the openly available Amazon review data set.

The project was part of the Cogito NLP workshop Spring 2020, with [Strise](https://github.com/strise/cogito-workshop-spring-2020).

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

------

Made with :heart: and python
