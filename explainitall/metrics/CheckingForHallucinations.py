import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util


def sim(embedding_1, embedding_2):
    s = util.pytorch_cos_sim(embedding_1, embedding_2)
    return s.detach().numpy()


class RAGHallucinationsChecker:
    def __init__(self, sbert_model: SentenceTransformer):
        nltk.download('punkt')
        self.sbert_model = sbert_model

    def load_doc(self, text, block_size=3):
        seqs = sent_tokenize(text, language='russian')
        count_block = max([1, len(seqs) // block_size])
        seqs = [list(x) for x in np.array_split(seqs, count_block)]
        snippets = [' '.join(x) for x in seqs]

        return snippets

    def get_support_seq(self, doc_snp, ans, prob=0.6, top_k=1):

        docs = []
        sn_ans = self.load_doc(ans, 1)

        for d in doc_snp:
            docs += self.load_doc(d, 1)

        top_k = min(top_k, len(docs))

        ans_v = self.sbert_model.encode(sn_ans)
        doc_v = self.sbert_model.encode(docs)
        matrix_ = sim(doc_v, ans_v)

        res = []
        for i in range(matrix_.shape[1]):
            slice_ = matrix_[:, i]
            top_indexes = np.argpartition(slice_, -top_k)[-top_k:]
            top_probs = matrix_[top_indexes, i]
            top_indexes[top_probs < prob] = -1

            reference_texts = []
            indexes = []
            for j, d in enumerate(top_indexes):
                if d >= 0:
                    reference_texts += [docs[d]]
                    indexes += [d]

            if len(indexes) > 0:
                res.append({'answer': sn_ans[i], 'reference_texts': reference_texts, 'indexes': indexes})

        return res

    def get_conf(self, doc_snp, ans, prob=0.6):
        answer_a = self.get_support_seq(ans=ans, doc_snp=doc_snp, prob=prob)
        len_all = len(ans)
        len_with_out_h = len(answer_a) - 1  # Учитываем пробелы

        for s in answer_a:
            len_with_out_h += len(s['answer'])

        return len_with_out_h / len_all

    def get_hallucinations_prob(self, doc_snp, ans, prob=0.6):
        return 1 - self.get_conf(doc_snp, ans, prob)
