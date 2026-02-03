import json
from datetime import datetime
import os

class Parameter:
    __instance = None  # Singleton to keep a consistent timestamp across instances

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Parameter, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self.__ExpName = None
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.__json_file = os.path.join(base_path, "Parameter.json")
        self.__history_header = None
        self.__origination_header = None
        self.__history_selected_header = None
        self.__jump_number = None
        self.__predict_month = None
        self.__split = None
        self.__verbose = None
        self.__random_seed = None
        self.__window_size = None
        self.__data_set = None
        self.__Nrows = None
        self.__Normalize = None
        self.__Time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.__InsideDataFilename = None
        self.__Epochs = None
        self.__BatchSize = None
        self.__EvaluationModelNames = None
        self.__EvaluationTimes = None
        self.__EnableSHAP = None
        self.__SHAPModelNames = None

        self.load(self.__json_file)

    def load(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.__history_header = data.get('HistoryHeader')
        self.__origination_header = data.get('OriginationHeader')
        self.__history_selected_header = data.get('HistorySelectedHeader')
        self.__jump_number = data.get('JumpNumber')
        self.__predict_month = data.get('PredictMonth')
        self.__split = data.get('Split')
        self.__verbose = data.get('Verbose')
        self.__random_seed = data.get('RandomSeed')
        self.__window_size = data.get('WindowSize')
        self.__data_set = data.get('DataSet')
        self.__Nrows = data.get('Nrows')
        self.__Normalize = data.get('Normalize')
        self.__ExpName = data.get('ExpName')
        self.__InsideDataFilename = data.get('InsideDataFilename')
        self.__Epochs = data.get('Epochs')
        self.__BatchSize = data.get('BatchSize')
        self.__EvaluationModelNames = data.get('EvaluationModelNames')
        self.__EvaluationTimes = data.get('EvaluationTimes')
        self.__EnableSHAP = data.get('EnableSHAP')
        self.__SHAPModelNames = data.get('SHAPModelNames')

    def to_dict(self):
        return {
            "ExpName": self.__ExpName,
            "HistoryHeader": self.__history_header,
            "OriginationHeader": self.__origination_header,
            "HistorySelectedHeader": self.__history_selected_header,
            "JumpNumber": self.__jump_number,
            "PredictMonth": self.__predict_month,
            "Split": self.__split,
            "Verbose": self.__verbose,
            "RandomSeed": self.__random_seed,
            "WindowSize": self.__window_size,
            "DataSet": self.__data_set,
            "Nrows": self.__Nrows,
            "Normalize": self.__Normalize,
            "InsideDataFilename": self.__InsideDataFilename,
            "Epochs": self.__Epochs,
            "BatchSize": self.__BatchSize,
            "EvaluationModelNames": self.__EvaluationModelNames,
            "EvaluationTimes": self.__EvaluationTimes,
            "EnableSHAP": self.__EnableSHAP,
            "SHAPModelNames": self.__SHAPModelNames,
        }

    def save(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    # Properties
    @property
    def ExpName(self):
        return self.__ExpName

    @ExpName.setter
    def ExpName(self, value):
        self.__ExpName = value

    @property
    def HistoryHeader(self):
        return self.__history_header

    @HistoryHeader.setter
    def HistoryHeader(self, value):
        self.__history_header = value

    @property
    def OriginationHeader(self):
        return self.__origination_header

    @OriginationHeader.setter
    def OriginationHeader(self, value):
        self.__origination_header = value

    @property
    def HistorySelectedHeader(self):
        return self.__history_selected_header

    @HistorySelectedHeader.setter
    def HistorySelectedHeader(self, value):
        self.__history_selected_header = value

    @property
    def JumpNumber(self):
        return self.__jump_number

    @JumpNumber.setter
    def JumpNumber(self, value):
        self.__jump_number = value

    @property
    def PredictMonth(self):
        return self.__predict_month

    @PredictMonth.setter
    def PredictMonth(self, value):
        self.__predict_month = value

    @property
    def Split(self):
        return self.__split

    @Split.setter
    def Split(self, value):
        self.__split = value

    @property
    def Verbose(self):
        return self.__verbose

    @Verbose.setter
    def Verbose(self, value):
        self.__verbose = value

    @property
    def RandomSeed(self):
        return self.__random_seed

    @RandomSeed.setter
    def RandomSeed(self, value):
        self.__random_seed = value

    @property
    def WindowSize(self):
        return self.__window_size

    @WindowSize.setter
    def WindowSize(self, value):
        self.__window_size = value

    @property
    def DataSet(self):
        return self.__data_set

    @DataSet.setter
    def DataSet(self, value):
        self.__data_set = value

    @property
    def Nrows(self):
        return self.__Nrows

    @Nrows.setter
    def Nrows(self, value):
        self.__Nrows = value

    @property
    def Time(self):
        return self.__Time

    @property
    def Normalize(self):
        return self.__Normalize

    @Normalize.setter
    def Normalize(self, value):
        self.__Normalize = value

    @property
    def InsideDataFilename(self):
        if self.__InsideDataFilename == "":
            return None
        return self.__InsideDataFilename

    @InsideDataFilename.setter
    def InsideDataFilename(self, value):
        self.__InsideDataFilename = value

    @property
    def Epochs(self):
        return self.__Epochs

    @Epochs.setter
    def Epochs(self, value):
        self.__Epochs = value

    @property
    def BatchSize(self):
        return self.__BatchSize

    @BatchSize.setter
    def BatchSize(self, value):
        self.__BatchSize = value

    @property
    def EvaluationModelNames(self):
        return self.__EvaluationModelNames

    @EvaluationModelNames.setter
    def EvaluationModelNames(self, value):
        self.__EvaluationModelNames = value

    @property
    def EvaluationTimes(self):
        return self.__EvaluationTimes

    @EvaluationTimes.setter
    def EvaluationTimes(self, value):
        self.__EvaluationTimes = value

    @property
    def EnableSHAP(self):
        return self.__EnableSHAP

    @EnableSHAP.setter
    def EnableSHAP(self, value):
        self.__EnableSHAP = value

    @property
    def SHAPModelNames(self):
        return self.__SHAPModelNames

    @SHAPModelNames.setter
    def SHAPModelNames(self, value):
        self.__SHAPModelNames = value
