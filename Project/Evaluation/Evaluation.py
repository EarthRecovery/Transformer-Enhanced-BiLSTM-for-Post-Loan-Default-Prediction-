from Paramaters.Parameter import Parameter
from Utils.EvaluationStructure import EvaluationStructure
import os
import importlib
import numpy as np
import json
from Explainability.SHAPExplainer import SHAPExplainer


class Evaluation:
    def __init__(self):
        self._parameter = Parameter()
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)
        if self._parameter.InsideDataFilename is not None:
            self._data_folder = os.path.join(parent_path, "data", self._parameter.InsideDataFilename)
        else:
            self._data_folder = os.path.join(parent_path, "data", f"{self._parameter.ExpName}-{self._parameter.Time}")

        self._model_list = []
        self._evaluation_list = [[] for _ in range(len(self._parameter.EvaluationModelNames))]

        self.__trainX = np.load(os.path.join(self._data_folder, "trainX.npy"), allow_pickle=True)
        self.__trainY = np.load(os.path.join(self._data_folder, "trainY.npy"), allow_pickle=True)
        self.__testX = np.load(os.path.join(self._data_folder, "testX.npy"), allow_pickle=True)
        self.__testY = np.load(os.path.join(self._data_folder, "testY.npy"), allow_pickle=True)

        self.__trainX = np.array(self.__trainX, dtype=np.float32)
        self.__trainY = np.array(self.__trainY, dtype=np.float32)
        self.__testX = np.array(self.__testX, dtype=np.float32)
        self.__testY = np.array(self.__testY, dtype=np.float32)

    def load_models(self):
        for model_name in self._parameter.EvaluationModelNames:
            module = importlib.import_module(f"Models.{model_name}")
            model_class = getattr(module, model_name)
            model_instance = model_class()
            self._model_list.append(model_instance)

    def train_models(self):
        for i in range(len(self._model_list)):
            self._model_list[i].train(self.__trainX, self.__trainY)
            print(f"Model name {self._model_list[i].model_name} trained successfully ( {i} / {len(self._model_list)}).")

    def test_and_evaluate_models(self):
        for i in range(len(self._model_list)):
            accuracy, precision, recall, binary_f1, auc, conf_matrix = self._model_list[i].evaluate_model_and_return_metrics(self.__testX, self.__testY)
            evaluation = EvaluationStructure(accuracy, precision, recall, binary_f1, auc, conf_matrix)
            self._evaluation_list[i].append(evaluation)
            print(f"Model name {self._model_list[i].model_name} evaluated successfully ({i}/{len(self._model_list)}).")

    def full_evaluation(self):
        for i in range(self._parameter.EvaluationTimes):
            self._model_list = []
            print(f"Evaluation {i + 1} / {self._parameter.EvaluationTimes}")
            self.load_models()
            self.train_models()
            self.test_and_evaluate_models()
            self.save_all_records()

    def run_shap(self, max_background=200, max_test=200, seed=42):
        if not self._parameter.EnableSHAP:
            print("SHAP is disabled. Skipping SHAP explainability.")
            return

        model_list = self._model_list
        if self._parameter.SHAPModelNames:
            shap_set = set(self._parameter.SHAPModelNames)
            model_list = [
                m for m in self._model_list
                if getattr(m, "model_name", None) in shap_set
            ]
            if not model_list:
                print("No matching models for SHAP. Skipping.")
                return

        explainer = SHAPExplainer(
            self._data_folder,
            self.__trainX,
            self.__trainY,
            self.__testX,
            self.__testY,
            self._parameter,
        )
        explainer.explain_models(
            model_list,
            max_background=max_background,
            max_test=max_test,
            seed=seed,
        )

    def save_all_records(self):
        all_results = {}

        # Traverse each model
        for model_idx, model_results in enumerate(self._evaluation_list):
            model_name = self._parameter.EvaluationModelNames[model_idx]

            # Collect all experiment results for each model
            all_results[model_name] = []

            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_binary_f1 = 0
            total_auc = 0
            num_experiments = len(model_results)
            summed_conf_matrix = None

            for experiment_idx, result in enumerate(model_results):
                # Convert EvaluationStructure to dict
                result_dict = {
                    "Accuracy": result.accuracy,
                    "Precision": result.precision,
                    "Recall": result.recall,
                    "BinaryF1": result.binary_f1,
                    "AUC": result.auc,
                    "ConfusionMatrix": result.conf_matrix.tolist() if hasattr(result.conf_matrix, 'tolist') else result.conf_matrix,
                }
                all_results[model_name].append(result_dict)

                conf_matrix_array = np.array(result.conf_matrix)
                if summed_conf_matrix is None:
                    summed_conf_matrix = conf_matrix_array
                else:
                    summed_conf_matrix += conf_matrix_array

                total_accuracy += result.accuracy
                total_precision += result.precision
                total_recall += result.recall
                total_binary_f1 += result.binary_f1
                total_auc += result.auc

            if num_experiments > 0:
                avg_result = {
                    "Accuracy": total_accuracy / num_experiments,
                    "Precision": total_precision / num_experiments,
                    "Recall": total_recall / num_experiments,
                    "BinaryF1": total_binary_f1 / num_experiments,
                    "AUC": total_auc / num_experiments,
                    "ConfusionMatrix": summed_conf_matrix.tolist() if summed_conf_matrix is not None else None,
                }

                all_results[model_name].append({"Average": avg_result})

        # Write to file
        record_path = os.path.join(self._data_folder, "all_results.json")
        with open(record_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"All experiment results saved to {record_path}")
