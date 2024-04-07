import numpy as np
from numpy.linalg import norm
from sentence_transformers import util


def sen_a(model, texts):
    embeddings = model.encode(texts)
    s = util.pytorch_cos_sim(embeddings, embeddings)
    s[s < 0] = 0
    matrix = (1 - s).to('cpu').detach().numpy()
    return matrix[matrix > 1e-7].mean()


class CosRelu:
    @staticmethod
    def dot_(v1, v2):
        """Скалярное произведение (у numpy не точно считает)"""
        sum_ = 0
        for i, v in enumerate(v2):
            sum_ += v1[i] * v
        return sum_

    @staticmethod
    def cos(v1, v2):
        """Косинус"""
        return CosRelu.dot_(v1, v2) / (norm(v1) * norm(v2))

    @staticmethod
    def cos_relu(y_orig, y_without_ner) -> float:
        """Рассчет ReLU от косинуса между векторам эмбеддингов"""
        r = CosRelu.cos(y_orig, y_without_ner)
        r = r if r >= 0 else 0
        return r


class ModelInterp:

    def __init__(self, model):
        self.model = model
        self.mask_token = model.tokenizer.mask_token

    def seq_interp(self, sent):
        refer = self.model.encode(sent)

        words = sent.split(' ')
        imp = []
        y_mod = []

        for k in range(len(words)):
            repl_word = ' '.join([word if i != k else self.mask_token for i, word in enumerate(words)])
            y_mod.append(repl_word)

        targ = self.model.encode(y_mod)

        for vect in targ:
            imp.append(1 - CosRelu.cos_relu(refer, vect))

        imp = np.array(imp)
        imp_sum = imp.sum()
        imp /= imp_sum

        return {'imp': imp, 'words': words}

    def dataset_interp(self, texts):
        ret_data = []
        for text in texts:
            interp_seq = self.seq_interp(text)
            ret_data.append(interp_seq)

        return ret_data

    def __claster_energy(self, cluster_data):
        elements = cluster_data['elements']
        vectors = self.model.encode(elements)
        cl_energe = np.array([sum(vector ** 2) for vector in vectors])
        return {'name': cluster_data['name'], 'sensitivity': sen_a(self.model, elements), 'mean': cl_energe.mean()}

    def clusters_interp(self, clusters_data):
        return [self.__claster_energy(cluster_data) for cluster_data in clusters_data]
