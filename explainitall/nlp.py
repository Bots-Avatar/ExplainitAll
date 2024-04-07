from functools import lru_cache

import gensim
import pymorphy2


class WordProcessor:

    def __init__(self, gensim_nlp_embeddings: gensim.models.keyedvectors.KeyedVectors):
        self._morph = pymorphy2.MorphAnalyzer()
        self.gensim_nlp_embeddings = gensim_nlp_embeddings

    @lru_cache(maxsize=None)
    def get_clean_word(self, word: str):
        return word.lower().strip()

    # @lru_cache(maxsize=None)
    def get_embeddable_words_batch(self, words: list):
        cleaned = [self.get_clean_word(w) for w in words]
        embeddable = [self.get_embeddable_word_or_none(w) for w in cleaned if w]
        return [w for w in embeddable if w]

    @lru_cache(maxsize=None)
    def get_morph_or_none(self, word: str):
        morphed = self._morph.parse(word)
        known_words = [x for x in morphed if x.is_known]
        if not known_words:
            return None
        return known_words[0]

    @lru_cache(maxsize=None)
    def get_normal_form_or_none(self, word: str):
        clean_word = self.get_clean_word(word)
        if not clean_word:
            return None
        word_morph = self.get_morph_or_none(clean_word)
        if not word_morph:
            return None
        return word_morph.normal_form

    @lru_cache(maxsize=None)
    def get_grammeme_or_none(self, word: str):
        clean_word = self.get_clean_word(word)
        if not clean_word:
            return None
        word_morph = self.get_morph_or_none(clean_word)
        if not word_morph:
            return None
        tag = word_morph.tag.POS
        tag = 'VERB' if word_morph and tag == 'INFN' else tag
        return tag

    def get_embeddable_word_or_none(self, word: str):
        normal_form = self.get_normal_form_or_none(word)
        grammeme = self.get_grammeme_or_none(word)
        if not normal_form or not grammeme:
            return None
        word_tagged = f'{normal_form}_{grammeme}'
        if word_tagged not in self.gensim_nlp_embeddings:
            return None
        return word_tagged
