import re


def get_all_words_from_text(text):
    words = re.findall(r"[\w]+", text)
    return words
