import numpy as np
import operator
from mlclas.stats import UniversalMetrics


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


class BPMLLResults:
    def __init__(self, global_error):
        self.predictedLabels = []
        self.topRankedLabels = []
        self.outputs = []
        self.final_global_error = global_error

    def add(self, predict_set, top_label, output):
        self.predictedLabels.append(predict_set)
        self.topRankedLabels.append(top_label)
        self.outputs.append(output)


class ActivationFunction:
    @staticmethod
    def activate(_input):
        return 2 / (1 + np.exp(-2 * _input)) - 1

    @staticmethod
    def derivative(_input):
        return 1 - np.square(ActivationFunction.activate(_input))


class BPMLLMetrics(UniversalMetrics):
    """ Metrics design for BPMLL according to the original paper """
    def __init__(self, expected, result):
        self.sampleNum = len(expected)
        expectedLabels = [[int(i) for i in expected[j]] for j in range(len(expected))]

        # self.predictedLabels = result.predictedLabels
        super().__init__(expectedLabels, result.predictedLabels)

        self.topRankedLabels = result.topRankedLabels
        self.outputs = result.outputs
        self.possibleLabelNum = len(self.outputs[0])
        self.final_error = result.final_global_error

        self.ap_prepared = False
        self.ap = None
        self.rl_prepared = False
        self.rl = None

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
        if self.rl_prepared is True:
            return self.rl

        rloss_sum = 0
        ap_sum = 0
        for sample_index in range(self.sampleNum):
            unodered_part = []
            expected_num = len(self.expectedLabels[sample_index])

            sample_output = self.outputs[sample_index]
            output_dic = {}
            for output_index in range(self.possibleLabelNum):
                output_dic[output_index] = sample_output[output_index]

            sorted_output = sorted(output_dic.items(), key=operator.itemgetter(1), reverse=True)

            temp_count = 0
            times = 0
            for sorted_tuples in sorted_output:
                if times == expected_num:
                    break

                if sorted_tuples[0] not in self.expectedLabels[sample_index]:
                    temp_count += 1
                else:
                    unodered_part.append(temp_count)
                    temp_count = 0
                    times += 1

            if len(unodered_part) != expected_num:
                raise Exception("function error for RankingLoss")

            pairs_num = 0
            fraction_sum = 0
            fraction_divide = 0
            for cal_index in range(expected_num):
                pairs_num += unodered_part[cal_index] * (expected_num - cal_index)
                # prepare for calculating average precision
                fraction_divide += unodered_part[cal_index] + 1
                fraction_sum += (cal_index + 1) / fraction_divide

            rloss_sum += pairs_num / (expected_num * (self.possibleLabelNum - expected_num))
            ap_sum += fraction_sum / expected_num

        self.ap = ap_sum / self.sampleNum
        self.rl = rloss_sum / self.sampleNum
        self.ap_prepared = True
        self.rl_prepared = True

        return self.rl

    def average_precision(self):
        # contained in the ranking_loss function to save running time
        if self.ap_prepared is True:
            return self.ap
        else:
            self.ranking_loss()
            return self.ap
