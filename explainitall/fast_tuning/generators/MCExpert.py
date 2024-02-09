from abc import ABCMeta

from explainitall.fast_tuning.ExpertBase import ExpertModel


class MCExpert(ExpertModel, metaclass=ABCMeta):

    def __init__(self, mc):
        self.mc = mc

    def get_bias(self, tokens):
        start = [1, 1] + tokens
        return self.mc.get_bias(*start[-2:])