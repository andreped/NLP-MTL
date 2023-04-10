from typing import List, Tuple

import nltk


class Review:
    def __init__(self, review_id: str, product_id: str, age: int, title: str, review_text: str, rating: int,
                 recommends: bool, upvotes: int, division_name: str, department_name: str, class_name: str):
        self.review_id = review_id
        self.product_id = product_id
        self.age = age
        self.title = title
        self.review_text = review_text
        self.rating = rating
        self.recommends = recommends
        self.upvotes = upvotes
        self.division_name = division_name
        self.department_name = department_name
        self.class_name = class_name

    def full_text(self) -> str:
        return f"{self.title} {self.review_text}"

    def words(self) -> List[str]:
        return nltk.word_tokenize(self.full_text())

    def tagged_words(self) -> List[Tuple[str, str]]:
        """ Returns each word in `full_text()` along with its predicted POS-tag. For instance, if `full_text()` is
            "I love this red shirt", the returned list will be:
            [('I', 'PRP'), ('love', 'VBP'), ('this', 'DT'), ('red', 'JJ'), ('shirt', 'NN')]
            where each of the tags refer to a specific grammatical class.
            See: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/ for a list of all
            possible tags"""
        return nltk.pos_tag(self.words())
