import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add the two parent directories to sys.path

from Algorithm.DataProcessing import DataProcessing
from Paramaters.Parameter import Parameter
from Algorithm.TrainTestSplit import TrainTestSplit
from Logs.Log import Log
from Evaluation.Evaluation import Evaluation


class Main:
    def __init__(self):
        self._parameter = Parameter()
        # self._parameter.InsideDataFilename = "Newtest"
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)
        if self._parameter.InsideDataFilename is not None:
            save_folder = os.path.join(parent_path, "data", self._parameter.InsideDataFilename)
        else:
            save_folder = os.path.join(parent_path, "data", f"{self._parameter.ExpName}-{self._parameter.Time}")
        os.makedirs(save_folder, exist_ok=True)
        self._log = Log()
        self._log.start_logging_cmd()

    def main_logic(self):
        # 1. Data preprocessing
        data_processing = DataProcessing()
        data_processing.load_data()

        # 2. Train/test split
        train_test_split = TrainTestSplit()
        train_test_split.split()

        # 3. Model training and evaluation
        evaluation = Evaluation()
        evaluation.full_evaluation()
        evaluation.run_shap()

        # 4. Logging
        self._log.log_code()

    def main(self):
        self.main_logic()
        self._log.stop_logging_cmd()
        self._log.log_code()


if __name__ == "__main__":
    main = Main()
    main.main()
    del main
