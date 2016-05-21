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
