from sklearn.preprocessing import MinMaxScaler
from Paramaters.Parameter import Parameter
import pandas as pd
import os


class DataProcessing:
    def __init__(self):
        self.__parameter = Parameter()
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)
        grandparent_path = os.path.dirname(parent_path)
        self.__dataset_name = os.path.join(grandparent_path, "Data", self.__parameter.DataSet)

    def load_data(self):
        dataset = pd.read_csv(
            self.__dataset_name,
            sep='|',
            header=None,
            names=self.__parameter.HistoryHeader.split(","),
            index_col=False,
            nrows=self.__parameter.Nrows,
            low_memory=False,
        )

        dataset = dataset[self.__parameter.HistorySelectedHeader.split(",")]

        # Normalize delinquency status: 0/1/2 (or '0'/'1'/'2') -> 0, others -> 1
        # Columns that are not processed before this step: loan id, reporting month, current UPB
        dataset['CURRENT LOAN DELINQUENCY STATUS'] = dataset['CURRENT LOAN DELINQUENCY STATUS'].replace(['0', '1', '2', 0, 1, 2], 0)
        dataset['CURRENT LOAN DELINQUENCY STATUS'] = dataset['CURRENT LOAN DELINQUENCY STATUS'].apply(lambda x: 1 if x != 0 else 0)

        # Columns not transformed before this step: legal maturity date, loan age
        # Add a new column: repayment stage = loan age / (remaining months + loan age)
        dataset['REPAYMENT_STAGE'] = dataset['LOAN AGE'] / (dataset['REMAINING MONTHS TO LEGAL MATURITY'] + dataset['LOAN AGE'])

        # One-hot encode MODIFICATION FLAG: Y / P / Null
        dataset['MODIFICATION FLAG'] = dataset['MODIFICATION FLAG'].fillna('Null')
        dataset['MODIFICATION FLAG'] = pd.Categorical(
            dataset['MODIFICATION FLAG'], categories=['Null', 'Y', 'P'], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['MODIFICATION FLAG'], prefix='MODIFICATION_FLAG', drop_first=True)

        # CURRENT DEFERRED UPB is not transformed before this step
        dataset['MODIFICATION COST'] = dataset['MODIFICATION COST'].fillna(0)

        # One-hot encode STEP MODIFICATION FLAG: Null / Y / N
        dataset['STEP MODIFICATION FLAG'] = dataset['STEP MODIFICATION FLAG'].fillna('Null')
        dataset['STEP MODIFICATION FLAG'] = pd.Categorical(
            dataset['STEP MODIFICATION FLAG'], categories=['Null', 'Y', 'N'], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['STEP MODIFICATION FLAG'], prefix='STEP_MODIFICATION_FLAG', drop_first=True)

        # One-hot encode DEFERRED PAYMENT PLAN: Null / P / Y
        dataset['DEFERRED PAYMENT PLAN'] = dataset['DEFERRED PAYMENT PLAN'].fillna('Null')
        dataset['DEFERRED PAYMENT PLAN'] = pd.Categorical(
            dataset['DEFERRED PAYMENT PLAN'], categories=['Null', 'P', 'Y'], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['DEFERRED PAYMENT PLAN'], prefix='DEFERRED_PAYMENT_PLAN', drop_first=True)

        # Create ELTV masked flag: < 2017-04 -> 0, otherwise 1
        dataset['ELTV_MASKED'] = dataset['MONTHLY REPORTING PERIOD'].apply(lambda x: 1 if x >= 201704 else 0)

        # Fill ELTV missing values
        dataset['ESTIMATED LOAN TO VALUE (ELTV)'] = dataset['ESTIMATED LOAN TO VALUE (ELTV)'].fillna(0)

        # Encode DELINQUENCY DUE TO DISASTER: null -> False, otherwise True
        dataset['DELINQUENCY DUE TO DISASTER'] = dataset['DELINQUENCY DUE TO DISASTER'].fillna('Null')
        dataset['DELINQUENCY DUE TO DISASTER'] = pd.Categorical(
            dataset['DELINQUENCY DUE TO DISASTER'], categories=['Null', 'Y'], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['DELINQUENCY DUE TO DISASTER'], prefix='DELINQUENCY_DUE_TO_DISASTER', drop_first=True)

        # Encode BORROWER ASSISTANCE STATUS CODE
        dataset['BORROWER ASSISTANCE STATUS CODE'] = dataset['BORROWER ASSISTANCE STATUS CODE'].fillna('Null')
        dataset['BORROWER ASSISTANCE STATUS CODE'] = pd.Categorical(
            dataset['BORROWER ASSISTANCE STATUS CODE'], categories=['Null', 'F', 'R', 'T'], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['BORROWER ASSISTANCE STATUS CODE'], prefix='BORROWER_ASSISTANCE_STATUS_CODE', drop_first=True)

        # Fill current month modification cost
        dataset['CURRENT MONTH MODIFICATION COST'] = dataset['CURRENT MONTH MODIFICATION COST'].fillna(0)

        # INTEREST BEARING UPB delta
        dataset['INTEREST BEARING UPB'] = dataset['INTEREST BEARING UPB'].astype('float32')
        dataset['INTEREST BEARING UPB-Delta'] = dataset.groupby('LOAN SEQUENCE NUMBER')['INTEREST BEARING UPB'].diff()
        dataset['INTEREST BEARING UPB-Delta'] = dataset['INTEREST BEARING UPB-Delta'].fillna(0)
        dataset['INTEREST BEARING UPB-Delta'] = dataset['INTEREST BEARING UPB-Delta'].abs()

        # Encode INTEREST BEARING UPB as 0 vs non-zero
        dataset['INTEREST BEARING UPB'] = dataset['INTEREST BEARING UPB'].apply(lambda x: 0.0 if x == 0.0 else 1.0)
        dataset['INTEREST BEARING UPB'] = pd.Categorical(
            dataset['INTEREST BEARING UPB'], categories=[0.0, 1.0], ordered=True)
        dataset = pd.get_dummies(dataset, columns=['INTEREST BEARING UPB'], prefix='INTEREST_BEARING_UPB', drop_first=True)

        # Monthly repayment delta
        dataset['CURRENT ACTUAL UPB'] = dataset['CURRENT ACTUAL UPB'].astype('float32')
        dataset['CURRENT ACTUAL UPB-Delta'] = dataset.groupby('LOAN SEQUENCE NUMBER')['CURRENT ACTUAL UPB'].diff()
        dataset['CURRENT ACTUAL UPB-Delta'] = dataset['CURRENT ACTUAL UPB-Delta'].fillna(0)
        dataset['CURRENT ACTUAL UPB-Delta'] = dataset['CURRENT ACTUAL UPB-Delta'].abs()

        print(dataset.columns)

        # Feature scaling
        if self.__parameter.Normalize is True:
            scaler = MinMaxScaler()
            dataset.iloc[:, 1:] = scaler.fit_transform(dataset.iloc[:, 1:])

        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)

        if self.__parameter.InsideDataFilename is not None:
            save_folder = os.path.join(parent_path, "data", self.__parameter.InsideDataFilename)
        else:
            save_folder = os.path.join(parent_path, "data", f"{self.__parameter.ExpName}-{self.__parameter.Time}")

        csv_name = os.path.join(
            save_folder,
            "encoded_MonthlyData.csv"
        )

        dataset.to_csv(csv_name, index=False)
        print("Encoded dataset saved to: ", csv_name)
