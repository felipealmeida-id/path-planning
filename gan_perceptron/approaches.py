import numpy as np
from enum import Enum


# Define approaches with enum to be easily used later
class WeightApproachEnum(Enum):
    LINEAR = "linear"
    EXPONENTIAL_SIGMOID = "exponential_sigmoid"
    CONSTANT = "constant"


class WeightApproach:
    # singleton instance
    __instance = None

    def __init__(self, approach, EPOCHS):
        if WeightApproach.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            WeightApproach.__instance = self
            self.approach = approach
            self.EPOCHS = EPOCHS

    @staticmethod
    def get_instance():
        from env_parser import Env

        if WeightApproach.__instance is None:
            WeightApproach(
                Env.get_instance().WEIGHT_APPROACH, Env.get_instance().EPOCHS
            )
        return WeightApproach.__instance

    def get_weights(self, epoch):
        if self.approach == WeightApproachEnum.LINEAR:
            return self.__linear(epoch)
        elif self.approach == WeightApproachEnum.EXPONENTIAL_SIGMOID:
            return self.__exponential_sigmoid(epoch)
        elif self.approach == WeightApproachEnum.CONSTANT:
            return self.__constant(epoch)
        else:
            raise Exception("Invalid approach")

    def __constant(self, epoch):
        evalWeight = 0.6
        regularWeight = 1 - evalWeight
        return evalWeight, regularWeight

    def __linear(self, epoch):
        evalWeight = epoch / self.EPOCHS
        regularWeight = (self.EPOCHS - epoch) / self.EPOCHS
        return evalWeight, regularWeight

    def __exponential_sigmoid(self, epoch):
        steepness = 0.02  # adjust this value to make transition more/less gradual
        sigmoid_epochs = 1 / (
            1 + np.exp(-steepness * (epoch - self.EPOCHS / 2))
        )  # Shifted & steepness adjusted sigmoid

        evalWeight = sigmoid_epochs
        regularWeight = 1 - sigmoid_epochs
        return evalWeight, regularWeight


class EvaluatorModuleApproachEnum(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


class EvaluatorModuleApproach:
    # singleton instance
    __instance = None

    def __init__(self, approach, EPOCHS):
        if EvaluatorModuleApproach.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EvaluatorModuleApproach.__instance = self
            self.approach = approach
            self.EPOCHS = EPOCHS

    @staticmethod
    def get_instance():
        from env_parser import Env

        if EvaluatorModuleApproach.__instance is None:
            EvaluatorModuleApproach(
                Env.get_instance().EVALUATOR_APPROACH, Env.get_instance().EPOCHS
            )
        return EvaluatorModuleApproach.__instance

    def get_evaluator_modules(self, epoch):
        if self.approach == EvaluatorModuleApproachEnum.CONSTANT:
            return self.__constant(epoch)
        elif self.approach == EvaluatorModuleApproachEnum.LINEAR:
            return self.__linear(epoch)
        else:
            raise Exception("Invalid approach")

    def __constant(self, epoch):
        from enums import EvaluatorModules

        return [EvaluatorModules.OUTOFBOUND]

    def __linear(self, epoch):
        from enums import EvaluatorModules

        if epoch < self.EPOCHS / 2:
            return [EvaluatorModules.OUTOFBOUND]
        else:
            return [EvaluatorModules.OUTOFBOUND, EvaluatorModules.BATTERY]
