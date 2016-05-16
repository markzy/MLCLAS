import copy
import numpy as np
import sys
import random
import math
import operator


class Nomalizer:
    def __init__(self, X_array, min_value, max_value):
        self.X_array = X_array
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self):
        if np.issubdtype(self.X_array.dtype, int):
            X_array = self.X_array.astype('float32')
        else:
            X_array = copy.copy(self.X_array)

        samples, features = X_array.shape
        for i in range(features):
            array_min = np.min(X_array[:, i])
            array_max = np.max(X_array[:, i])
            for j in range(samples):
                X_array[j, i] = ((X_array[j, i] - array_min) / (array_max - array_min) * (self.max_value - self.min_value)) + self.min_value
        return X_array


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
    def __init__(self, modelOutLabels, idealLabels):
        self.parameters = []
        self.build(modelOutLabels, idealLabels)

    def build(self, modelOutLabels, idealLabels):
        samples = len(idealLabels)
        labels = len(idealLabels[0])
        threshholds = [0 for i in range(samples)]

        if len(modelOutLabels) != samples or len(modelOutLabels[0]) != labels:
            raise Exception("inconsistent shape of two matrix")

        for i in range(samples):
            isLabelValue = [float('inf') for i in range(labels)]
            isNotLabelValue = [float('-inf') for i in range(labels)]
            for j in range(labels):
                if idealLabels[i][j] == 1:
                    isLabelValue[j] = modelOutLabels[i][j]
                else:
                    isNotLabelValue[j] = modelOutLabels[i][j]

            isLabelMin = min(isLabelValue)
            isNotLabelMax = max(isNotLabelValue)

            if isLabelMin != isNotLabelMax:
                if isLabelMin == sys.float_info.max:
                    threshholds[i] = isNotLabelMax + 0.1
                elif isNotLabelMax == sys.float_info.min:
                    threshholds[i] = isLabelMin - 0.1
                else:
                    threshholds[i] = (isLabelMin + isNotLabelMax) / 2
            else:
                threshholds[i] = isLabelMin

        modelOutLabels = np.concatenate((modelOutLabels, np.array([np.ones(samples)]).T), axis=1)
        self.parameters = np.linalg.lstsq(modelOutLabels, threshholds)[0]

    def computeThreshold(self, outputs):
        parameter_length = len(self.parameters)
        b_index = parameter_length - 1

        if len(outputs) != b_index:
            raise Exception('inconsistent length')

        threshold = 0
        for i in range(b_index):
            threshold += outputs[i] * self.parameters[i]
        threshold += self.parameters[b_index]

        return threshold


class BPMLLResults:
    def __init__(self, global_error):
        self.predictedLabels = []
        self.topRankedLabels = []
        self.outputs = []
        self.final_global_error = global_error

    def add(self, predictSet, topLabel, output):
        self.predictedLabels.append(predictSet)
        self.topRankedLabels.append(topLabel)
        self.outputs.append(output)


class ActivationFunction:
    @staticmethod
    def activate(_input):
        return 2 / (1 + np.exp(-2 * _input)) - 1

    @staticmethod
    def derivative(_input):
        return 1 - np.square(ActivationFunction.activate(_input))


class EvaluationMetrics:
    def __init__(self, expected, result):
        self.sampleNum = len(expected)

        self.expectedLabels = [[int(i) for i in expected[j]] for j in range(len(expected))]

        self.predictedLabels = result.predictedLabels
        self.topRankedLabels = result.topRankedLabels
        self.outputs = result.outputs
        self.possibleLabelNum = len(self.outputs[0])
        self.final_error = result.final_global_error

        self.ap_prepared = False
        self.ap = None

    def hamming_loss(self):
        diff_sum = 0
        for i in range(self.sampleNum):
            labels_sum = len(self.expectedLabels[i])
            intersection = 0
            for label in self.predictedLabels[i]:
                if label in self.expectedLabels[i]:
                    intersection += 1
            diff_sum += labels_sum - intersection

        return diff_sum / (self.possibleLabelNum * self.sampleNum)

    def one_error(self):
        error_sum = 0
        for i in range(self.sampleNum):
            if self.topRankedLabels[i] not in self.expectedLabels[i]:
                error_sum += 1

        return error_sum / self.sampleNum

    def coverage(self):
        cover_sum = 0
        for i in range(self.sampleNum):
            label_outputs = []
            for label in self.expectedLabels[i]:
                label_outputs.append(self.outputs[i][label])
            min_output = min(label_outputs)
            for j in range(self.possibleLabelNum):
                if self.outputs[i][j] >= min_output:
                    cover_sum += 1

        return (cover_sum / self.sampleNum) - 1

    def ranking_loss(self):
        rloss_sum = 0
        ap_sum = 0
        for sample_index in range(self.sampleNum):
            unoderedPart = []
            expectedNum = len(self.expectedLabels[sample_index])

            sample_output = self.outputs[sample_index]
            output_dic = {}
            for output_index in range(self.possibleLabelNum):
                output_dic[output_index] = sample_output[output_index]

            sorted_output = sorted(output_dic.items(), key=operator.itemgetter(1), reverse=True)

            temp_count = 0
            times = 0
            for sorted_tuples in sorted_output:
                if times == expectedNum:
                    break

                if sorted_tuples[0] not in self.expectedLabels[sample_index]:
                    temp_count += 1
                else:
                    unoderedPart.append(temp_count)
                    temp_count = 0
                    times += 1

            if len(unoderedPart) != expectedNum:
                raise Exception("function error for RankingLoss")

            pairs_num = 0
            fraction_sum = 0
            fraction_divide = 0
            for cal_index in range(expectedNum):
                pairs_num += unoderedPart[cal_index] * (expectedNum - cal_index)
                # prepare for calculating average precision
                fraction_divide += unoderedPart[cal_index] + 1
                fraction_sum += (cal_index + 1) / fraction_divide

            rloss_sum += pairs_num / (expectedNum * (self.possibleLabelNum - expectedNum))
            ap_sum += fraction_sum / expectedNum

        self.ap = ap_sum / self.sampleNum
        self.ap_prepared = True

        return rloss_sum / self.sampleNum

    def average_precision(self):
        if self.ap_prepared is True:
            return self.ap
        else:
            raise Exception('please run ranking_loss function first!')
