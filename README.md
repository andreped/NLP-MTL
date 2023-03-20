# NLP-MTL
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

Code from a workshop of which I trained a deep neural network to solve multiple classification tasks simultaneously from free text. For this I used multi-task learning (MTL), using the openly available Amazon review data set. The code was made in a couple of hours. Thus, no real effort to clean or refactor the code was made. However, it might be of use for someone who want to play around with Natural Language Processing (NLP).

The project was part of the Cogito NLP workshop Spring 2020, with [Strise](https://github.com/strise/cogito-workshop-spring-2020).

## How to use
Create a virtual environment, activate it, and install dependencies:
```
virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

Perform experiment by running the following lines:
```
cd cogito-nlp-workshop-spring-2020/
python train.py
```

## Troubleshooting
If any issues with numpy is observed when installing dependencies using the requirements.txt file, changing numpy version might help:
```
pip install numpy==1.16.0
```

------

Made with :heart: and python
