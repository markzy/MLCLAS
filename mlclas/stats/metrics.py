import operator


class Aggregate:
    @staticmethod
    def intersection(a, b):
        inter = 0
        for i in a:
            if i in b:
                inter += 1
        return inter

    @staticmethod
    def sum(a, b):
        return len(a) + len(b) - Aggregate.intersection(a, b)

    @staticmethod
    def sym_difference(a, b):
        return len(a) + len(b) - 2 * Aggregate.intersection(a, b)


class UniversalMetrics:
    def __init__(self, expected, predicted):
        self.sampleNum = len(expected)
        self.expectedLabels = [[int(i) for i in expected[j]] for j in range(len(expected))]
        # fix for divide by zero problems, this will not affect the final result
        for predict_index in range(len(predicted)):
            if len(predicted[predict_index]) == 0:
                predicted[predict_index].append(None)
        self.predictedLabels = predicted

    def accuracy(self):
        result = 0
        for index in range(self.sampleNum):
            expected = self.expectedLabels[index]
            predicted = self.predictedLabels[index]
            result += Aggregate.intersection(expected, predicted) / Aggregate.sum(expected, predicted)
        return result / self.sampleNum

    def precision(self):
        result = 0
        for index in range(self.sampleNum):
            expected = self.expectedLabels[index]
            predicted = self.predictedLabels[index]
            result += Aggregate.intersection(expected, predicted) / len(predicted)
        return result / self.sampleNum


class RankResults:
    def __init__(self):
        self.predictedLabels = []
        self.topRankedLabels = []
        self.outputs = []

    def add(self, predict_set, top_label, output):
        self.predictedLabels.append(predict_set)
        self.topRankedLabels.append(top_label)
        self.outputs.append(output)


class RankMetrics(UniversalMetrics):
    """ Metrics design for ranking systems"""

    def __init__(self, expected, result):
        self.sampleNum = len(expected)
        expectedLabels = [[int(i) for i in expected[j]] for j in range(len(expected))]

        super().__init__(expectedLabels, result.predictedLabels)

        self.topRankedLabels = result.topRankedLabels
        self.outputs = result.outputs
        self.possibleLabelNum = len(self.outputs[0])

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
