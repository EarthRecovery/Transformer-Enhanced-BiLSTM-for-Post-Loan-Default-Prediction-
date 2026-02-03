from Paramaters.Parameter import Parameter
from Logs.Logger import Logger
import os
import shutil as shutil2
import sys


class Log:
    def __init__(self):
        self._parameter = Parameter()
        base_path = os.path.dirname(os.path.abspath(__file__))
        self._parent_path = os.path.dirname(base_path)

        if self._parameter.InsideDataFilename is not None:
            self._data_folder = os.path.join(self._parent_path, "data", self._parameter.InsideDataFilename)
        else:
            self._data_folder = os.path.join(self._parent_path, "data", f"{self._parameter.ExpName}-{self._parameter.Time}")

        self._log_path = os.path.join(self._data_folder, "log.txt")
        if not os.path.exists(self._log_path):
            with open(self._log_path, 'w', encoding='utf-8') as f:
                f.write("")
        self._original_stdout = sys.stdout
        self._logger = None

    def start_logging_cmd(self):
        self._logger = Logger(self._log_path)
        sys.stdout = self._logger  # Redirect stdout to the log file

    def stop_logging_cmd(self):
        if self._logger:
            self._logger.close()
            sys.stdout = self._original_stdout
            print(f"Log file saved to {self._log_path}")

    def log_code(self):
        algorithm_path = os.path.join(self._parent_path, "Algorithm")
        main_path = os.path.join(self._parent_path, "Main")
        Parameter_path = os.path.join(self._parent_path, "Paramaters")
        Evaluation_path = os.path.join(self._parent_path, "Evaluation")
        Utils_path = os.path.join(self._parent_path, "Utils")

        if not os.path.exists(os.path.join(self._data_folder, "code")):
            os.makedirs(os.path.join(self._data_folder, "code"))
        code_folder_path = os.path.join(self._data_folder, "code")

        for filename in os.listdir(algorithm_path):
            if filename.endswith(".py"):
                # Create Algorithm folder
                if not os.path.exists(os.path.join(code_folder_path, "Algorithm")):
                    os.makedirs(os.path.join(code_folder_path, "Algorithm"))
                shutil2.copy(os.path.join(algorithm_path, filename), os.path.join(code_folder_path, "Algorithm", filename))

        for filename in os.listdir(main_path):
            if filename.endswith(".py"):
                if not os.path.exists(os.path.join(code_folder_path, "Main")):
                    os.makedirs(os.path.join(code_folder_path, "Main"))
                shutil2.copy(os.path.join(main_path, filename), os.path.join(code_folder_path, "Main", filename))

        for filename in os.listdir(Parameter_path):
            if filename.endswith(".py") or filename.endswith(".json"):
                if not os.path.exists(os.path.join(code_folder_path, "Paramaters")):
                    os.makedirs(os.path.join(code_folder_path, "Paramaters"))
                shutil2.copy(os.path.join(Parameter_path, filename), os.path.join(code_folder_path, "Paramaters", filename))

        for filename in os.listdir(Evaluation_path):
            if filename.endswith(".py"):
                if not os.path.exists(os.path.join(code_folder_path, "Evaluation")):
                    os.makedirs(os.path.join(code_folder_path, "Evaluation"))
                shutil2.copy(os.path.join(Evaluation_path, filename), os.path.join(code_folder_path, "Evaluation", filename))

        for filename in os.listdir(Utils_path):
            if filename.endswith(".py"):
                if not os.path.exists(os.path.join(code_folder_path, "Utils")):
                    os.makedirs(os.path.join(code_folder_path, "Utils"))
                shutil2.copy(os.path.join(Utils_path, filename), os.path.join(code_folder_path, "Utils", filename))

        for filename in os.listdir(Utils_path):
            if filename.endswith(".py"):
                if not os.path.exists(os.path.join(code_folder_path, "Models")):
                    os.makedirs(os.path.join(code_folder_path, "Models"))
                shutil2.copy(os.path.join(Utils_path, filename), os.path.join(code_folder_path, "Models", filename))
