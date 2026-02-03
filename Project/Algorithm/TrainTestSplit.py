from Paramaters.Parameter import Parameter
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class TrainTestSplit:
    def __init__(self):
        self.__parameter = Parameter()
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)

        if self.__parameter.InsideDataFilename is not None:
            self.__data_folder = os.path.join(parent_path, "data", self.__parameter.InsideDataFilename)
        else:
            self.__data_folder = os.path.join(parent_path, "data", f"{self.__parameter.ExpName}-{self.__parameter.Time}")

        self.EncodedDataset = pd.read_csv(
            os.path.join(self.__data_folder, "encoded_MonthlyData.csv"),
            sep=',',
            header=0,
            low_memory=False,
        )

    def split(self):
        # Split into two groups by borrower:
        # - delinquent_group: at least one status == 1
        # - non_delinquent_group: all status == 0
        sorted = self.EncodedDataset.sort_values('LOAN SEQUENCE NUMBER')
        grouped = sorted.groupby('LOAN SEQUENCE NUMBER')
        has_delinquency = grouped['CURRENT LOAN DELINQUENCY STATUS'].apply(lambda x: (x == 1).any())
        delinquent_ids = has_delinquency[has_delinquency].index
        non_delinquent_ids = has_delinquency[~has_delinquency].index
        delinquent_group = sorted[sorted['LOAN SEQUENCE NUMBER'].isin(delinquent_ids)]
        non_delinquent_group = sorted[sorted['LOAN SEQUENCE NUMBER'].isin(non_delinquent_ids)]
        delinquent_group = (
            delinquent_group
            .sort_values(['LOAN SEQUENCE NUMBER', 'REMAINING MONTHS TO LEGAL MATURITY'], ascending=[True, False])
            .reset_index(drop=True)
        )

        # Apply the same sorting for non_delinquent_group
        non_delinquent_group = (
            non_delinquent_group
            .sort_values(['LOAN SEQUENCE NUMBER', 'REMAINING MONTHS TO LEGAL MATURITY'], ascending=[True, False])
            .reset_index(drop=True)
        )

        # For delinquent_group: keep records up to and including the first STATUS == 1
        filtered_delinquent_group = []

        for loan_id, group in delinquent_group.groupby('LOAN SEQUENCE NUMBER'):
            # Find the first row where STATUS == 1 (global index)
            first_delinquency_index = group[group['CURRENT LOAN DELINQUENCY STATUS'] == 1].index.min()

            if pd.notna(first_delinquency_index):
                # Get the position within the group
                cutoff = group.index.get_loc(first_delinquency_index)
                # Keep rows from the start to the cutoff (inclusive)
                filtered = group.iloc[:cutoff + 1]
                filtered_delinquent_group.append(filtered)

        delinquent_group = pd.concat(filtered_delinquent_group).reset_index(drop=True)

        print(f"Filtered delinquent_group shape: {delinquent_group.shape}")

        # For delinquent_group: keep the last WINDOW_SIZE rows in each group; drop groups smaller than WINDOW_SIZE
        filtered_groups = []
        for loan_id, group in delinquent_group.groupby('LOAN SEQUENCE NUMBER'):
            if len(group) >= self.__parameter.WindowSize:
                # Keep the last WINDOW_SIZE rows
                filtered = group.tail(self.__parameter.WindowSize)
                filtered_groups.append(filtered)
        delinquent_group = pd.concat(filtered_groups).reset_index(drop=True)
        print(f"Filtered delinquent_group with WINDOW_SIZE={self.__parameter.WindowSize}, shape: {delinquent_group.shape}")

        # For delinquent_group: within each group, use first WINDOW_SIZE-PREDICT_MONTH as X and last PREDICT_MONTH as y
        X_list = []
        y_list = []

        count = 0
        for loan_id, group in delinquent_group.groupby('LOAN SEQUENCE NUMBER'):
            if len(group) == self.__parameter.WindowSize:
                count += 1
                group = group.reset_index(drop=True)

                # Input: first WINDOW_SIZE - PREDICT_MONTH rows
                X = group.iloc[:self.__parameter.WindowSize - self.__parameter.PredictMonth]

                # Labels: last PREDICT_MONTH rows
                y = group.iloc[-self.__parameter.PredictMonth:]

                X_list.append(X)
                y_list.append(y)

        # Merge all X and y into DataFrames
        X_delinquent = pd.concat(X_list).reset_index(drop=True)
        y_delinquent = pd.concat(y_list).reset_index(drop=True)

        print(f"Number of delinquent samples: {count}")

        # For non_delinquent_group: randomly choose count groups with length >= WINDOW_SIZE
        X_list = []
        y_list = []

        eligible_groups = [
            (loan_id, group)
            for loan_id, group in non_delinquent_group.groupby('LOAN SEQUENCE NUMBER')
            if len(group) >= self.__parameter.WindowSize
        ]

        # Check if there are enough eligible groups
        if len(eligible_groups) < count:
            raise ValueError(f"Not enough non-delinquent groups for {count}. Only {len(eligible_groups)} available.")

        # Randomly select count groups
        selected_groups = random.sample(eligible_groups, count)

        # Sort each selected group and reset index
        selected_groups = [
            (
                loan_id,
                group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False).reset_index(drop=True)
            )
            for loan_id, group in selected_groups
        ]

        for loan_id, group in selected_groups:
            start_idx = random.randint(0, len(group) - self.__parameter.WindowSize)
            selected_window = group.iloc[start_idx:start_idx + self.__parameter.WindowSize].reset_index(drop=True)

            # Input: first WINDOW_SIZE - PREDICT_MONTH rows
            X = selected_window.iloc[:self.__parameter.WindowSize - self.__parameter.PredictMonth]

            # Labels: last PREDICT_MONTH rows of STATUS
            y = selected_window.iloc[-self.__parameter.PredictMonth:]

            X_list.append(X)
            y_list.append(y)

        X_non_delinquent = pd.concat(X_list).reset_index(drop=True)
        y_non_delinquent = pd.concat(y_list).reset_index(drop=True)

        # Build training and test sets from X_delinquent/X_non_delinquent and labels
        X_delinquent_list = []
        for _, group in X_delinquent.groupby('LOAN SEQUENCE NUMBER'):
            group = group.drop(columns=['LOAN SEQUENCE NUMBER', 'CURRENT LOAN DELINQUENCY STATUS'])
            group_numpy = group.to_numpy(dtype=np.float32)
            X_delinquent_list.append(group_numpy)

        X_delinquent_list = np.array(X_delinquent_list)
        y_delinquent_list = np.ones(len(X_delinquent_list), dtype=int)
        print(X_delinquent_list.dtype)

        X_non_delinquent_list = []
        for _, group in X_non_delinquent.groupby('LOAN SEQUENCE NUMBER'):
            group = group.drop(columns=['LOAN SEQUENCE NUMBER', 'CURRENT LOAN DELINQUENCY STATUS'])
            group_numpy = group.to_numpy(dtype=np.float32)
            X_non_delinquent_list.append(group_numpy)

        X_non_delinquent_list = np.array(X_non_delinquent_list)
        y_non_delinquent_list = np.zeros(len(X_non_delinquent_list), dtype=int)

        # Train/test split
        X = np.concatenate((X_delinquent_list, X_non_delinquent_list), axis=0)
        y = np.concatenate((y_delinquent_list, y_non_delinquent_list), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
            shuffle=True,
        )

        # Save datasets
        np.save(os.path.join(self.__data_folder, "trainX.npy"), X_train)
        np.save(os.path.join(self.__data_folder, "trainY.npy"), y_train)
        np.save(os.path.join(self.__data_folder, "testX.npy"), X_test)
        np.save(os.path.join(self.__data_folder, "testY.npy"), y_test)
