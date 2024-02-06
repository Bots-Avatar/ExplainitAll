import re


def split_text_by_whitespaces(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = re.split(r"\s", text)
    return tokens
