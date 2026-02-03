class EvaluationStructure:
    def __init__(self, accuracy, precision, recall, binary_f1, auc, conf_matrix):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.binary_f1 = binary_f1
        self.auc = auc
        self.conf_matrix = conf_matrix

    