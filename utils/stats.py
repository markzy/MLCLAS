class Aggregate:
    @staticmethod
    def intersection(a,b):
        inter = 0
        for i in a:
            if i in b:
                inter += 1
        return inter

    @staticmethod
    def sum(a,b):
        return len(a) + len(b) - Aggregate.intersection(a, b)

    @staticmethod
    def symDifference(a,b):
        return len(a) + len(b) - 2 * Aggregate.intersection(a, b)

