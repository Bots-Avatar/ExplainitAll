import abc

class ExpertModel(abc.ABC):
    @abc.abstractmethod
    def get_bias(self, tokens):
        """Вычисление bias из вероятностной модели"""
    