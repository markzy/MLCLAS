
class UniversalMetrics:
    def __init__(self, classNum, expected, result):
        self.sampleNum = len(expected)
        self.classNum = classNum
        self.expectedLabels = [[int(i) for i in expected[j]] for j in range(len(expected))]
        self.predictedLabels = result

    def accuracy(self):
        result = 0
        for index in range(self.sampleNum):
            intersection = 0
            expected = self.expectedLabels[index]
            predicted = self.predictedLabels[index]
            for label in expected:
                if label in predicted:
                    intersection += 1

            union = len(expected) + len(predicted) - intersection
            result += intersection/union
        return result/self.sampleNum
