from abc import ABC, abstractmethod
from Paramaters.Parameter import Parameter
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import tensorflow as tf

class ModelInterface(ABC):

    def __init__(self):
        self._parameter = Parameter()
        self._epochs = self._parameter.Epochs
        self._batch_size = self._parameter.BatchSize
        self._verbose = self._parameter.Verbose

        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path) 

        if self._parameter.InsideDataFilename is not None:
            self._data_folder = os.path.join(parent_path, "data", self._parameter.InsideDataFilename)
        else:
            self._data_folder = os.path.join(parent_path, "data", f"{self._parameter.ExpName}-{self._parameter.Time}")

        self._model = None
        self.model_name = None

    @abstractmethod    
    def getModel(self,train_data, train_labels):
        pass

    def train(self, train_data, train_labels):   
        if self._model is None:
            self._model = self.getModel(train_data, train_labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=1024) \
                                    .cache() \
                                    .batch(self._batch_size) \
                                    .prefetch(tf.data.AUTOTUNE)
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model.fit(train_dataset, epochs=self._epochs, batch_size=self._batch_size, verbose=self._verbose)

    def evaluate_model_and_return_metrics(self, test_data, test_labels):
        y_pred_binary = []
        y_pred = self._model.predict(test_data)
        for i in y_pred: 
            if i >= 0.5:
                y_pred_binary.append(1)
            else:
                y_pred_binary.append(0)

        conf_matrix = confusion_matrix(test_labels, y_pred_binary)

        accuracy = accuracy_score(test_labels, y_pred_binary)
        precision = precision_score(test_labels, y_pred_binary)
        recall = recall_score(test_labels, y_pred_binary)
        binary_f1 = f1_score(test_labels, y_pred_binary)
        auc = roc_auc_score(test_labels, y_pred)

        return accuracy, precision, recall, binary_f1, auc, conf_matrix

    def predict(self, new_data):
        predictions = self._model.predict(new_data)
        return predictions
    
    # def record(self,conf_matrix, accuracy, precision, recall, binary_f1, auc):
    #     record_data = {
    #         "ConfusionMatrix": conf_matrix.tolist() if hasattr(conf_matrix, 'tolist') else conf_matrix,
    #         "Accuracy": accuracy,
    #         "Precision": precision,
    #         "Recall": recall,
    #         "BinaryF1": binary_f1,
    #         "AUC": auc,
    #         "Time": self._parameter.Time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(self._parameter.Time, 'strftime') else self._parameter.Time,
    #         "Experiment": self._parameter.ExpName
    #     }

    #     record_path = os.path.join(self._data_folder, "result.json")
    #     with open(record_path, 'w') as f:
    #         json.dump(record_data, f, indent=4)

    #     print(f"Experiment result saved to {record_path}")
