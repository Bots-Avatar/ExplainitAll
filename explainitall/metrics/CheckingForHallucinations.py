import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util


def sim_cosine(embedding_1, embedding_2):
    s = util.pytorch_cos_sim(embedding_1, embedding_2)
    return s.detach().numpy()

def sim_euclidean(embedding_1, embedding_2):
    s = np.linalg.norm(embedding_1 - embedding_2, axis=1)
    return s

def sim_manhattan(embedding_1, embedding_2):
    s = np.sum(np.abs(embedding_1 - embedding_2), axis=1)
    return s

def sim_cross_encoder(encoder, sentences1, sentences2):
    scores = encoder.predict([sentences1, sentences2])
    return scores

class RAGHallucinationsChecker:
    def __init__(self, sbert_model: SentenceTransformer, cross_encoder=None, language = 'russian'):
        nltk.download('punkt')
        self.sbert_model = sbert_model
        self.cross_encoder = cross_encoder
        self.language = language

    def load_doc(self, text, block_size=3):
        seqs = sent_tokenize(text, language=self.language)
        count_block = max([1, len(seqs) // block_size])
        seqs = [list(x) for x in np.array_split(seqs, count_block)]
        snippets = [' '.join(x) for x in seqs]

        return snippets

    def get_support_seq(self, doc_snp, ans, prob=0.6, top_k=1, sim_metric='cosine'):

        docs = []
        sn_ans = self.load_doc(ans, 1)

        for d in doc_snp:
            docs += self.load_doc(d, 1)

        top_k = min(top_k, len(docs))

        ans_v = self.sbert_model.encode(sn_ans)
        doc_v = self.sbert_model.encode(docs)

        if sim_metric == 'cosine':
            matrix_ = sim_cosine(doc_v, ans_v)
        elif sim_metric == 'euclidean':
            matrix_ = sim_euclidean(doc_v, ans_v)
        elif sim_metric == 'manhattan':
            matrix_ = sim_manhattan(doc_v, ans_v)
        elif sim_metric == 'cross_encoder' and self.cross_encoder is not None:
            matrix_ = sim_cross_encoder(self.cross_encoder, docs, sn_ans)
        else:
            raise ValueError("Unsupported similarity metric or cross-encoder is not provided.")

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

    def get_conf(self, doc_snp, ans, prob=0.6, sim_metric='cosine'):
        answer_a = self.get_support_seq(ans=ans, doc_snp=doc_snp, prob=prob, sim_metric=sim_metric)
        len_all = len(ans)
        len_with_out_h = len(answer_a) - 1  # Учитываем пробелы

        for s in answer_a:
            len_with_out_h += len(s['answer'])

        return len_with_out_h / len_all

    def get_hallucinations_prob(self, doc_snp, ans, prob=0.6, sim_metric='cosine'):
        return 1 - self.get_conf(doc_snp, ans, prob, sim_metric)

