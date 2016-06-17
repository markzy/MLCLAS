import numpy as np


class TrainPair:
    def __init__(self, attributes, labels):
        self.attributes = attributes
        self.labels = labels
        self.isLabel = []
        self.notLabel = []
        for j in range(labels.shape[0]):
            if labels[j] == 1:
                self.isLabel.append(j)
            else:
                self.notLabel.append(j)


class ThresholdFunction:
    """ Threshold Function built according to the original paper specified in the bpmll.py """

    def __init__(self, model_output, ideal_labels):
        self.parameters = []
        self.build(model_output, ideal_labels)

    def build(self, model_output, ideal_labels):
        samples = len(ideal_labels)
        labels = len(ideal_labels[0])
        threshholds = np.zeros(samples)

        if len(model_output) != samples or len(model_output[0]) != labels:
            raise Exception("inconsistent shape of two input matrix while building ThresholdFunction")

        for sample_index in range(samples):
            label_value = [float('inf') for i in range(labels)]
            notlabel_value = [float('-inf') for i in range(labels)]
            for j in range(labels):
                if ideal_labels[sample_index][j] == 1:
                    label_value[j] = model_output[sample_index][j]
                else:
                    notlabel_value[j] = model_output[sample_index][j]

            label_min = min(label_value)
            notlabel_max = max(notlabel_value)

            if label_min != notlabel_max:
                if label_min == float('inf'):
                    threshholds[sample_index] = notlabel_max + 0.1
                elif notlabel_max == float('-inf'):
                    threshholds[sample_index] = label_min - 0.1
                else:
                    threshholds[sample_index] = (label_min + notlabel_max) / 2
            else:
                threshholds[sample_index] = label_min

        model_output = np.concatenate((model_output, np.array([np.ones(samples)]).T), axis=1)
        self.parameters = np.linalg.lstsq(model_output, threshholds)[0]

    def compute_threshold(self, outputs):
        parameter_length = len(self.parameters)
        b_index = parameter_length - 1

        if len(outputs) != b_index:
            raise Exception('inconsistent output length with the trained array')

        threshold = 0
        for i in range(b_index):
            threshold += outputs[i] * self.parameters[i]
        threshold += self.parameters[b_index]

        return threshold


class ActivationFunction:
    @staticmethod
    def activate(_input):
        return 2 / (1 + np.exp(-2 * _input)) - 1

    @staticmethod
    def derivative(_input):
        return 1 - np.square(ActivationFunction.activate(_input))
