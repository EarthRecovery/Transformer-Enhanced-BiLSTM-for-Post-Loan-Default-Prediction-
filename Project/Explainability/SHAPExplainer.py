import os
import json
import numpy as np
import pandas as pd
from keras.layers import Input, Reshape
from keras.models import Model


class SHAPExplainer:
    def __init__(self, data_folder, trainX, trainY, testX, testY, parameter):
        self._data_folder = data_folder
        self._trainX = np.array(trainX, dtype=np.float32)
        self._trainY = np.array(trainY, dtype=np.float32)
        self._testX = np.array(testX, dtype=np.float32)
        self._testY = np.array(testY, dtype=np.float32)
        self._parameter = parameter

    def _flatten_time_features(self, data_3d):
        return np.reshape(data_3d, (data_3d.shape[0], data_3d.shape[1] * data_3d.shape[2]))

    def _sample_rows(self, data_2d, max_rows, seed):
        if data_2d.shape[0] <= max_rows:
            return data_2d
        rng = np.random.default_rng(seed)
        idx = rng.choice(data_2d.shape[0], size=max_rows, replace=False)
        return data_2d[idx]

    def _load_feature_names(self):
        csv_path = os.path.join(self._data_folder, "encoded_MonthlyData.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, nrows=1, low_memory=False)
        base_features = [
            c for c in df.columns
            if c not in ["LOAN SEQUENCE NUMBER", "CURRENT LOAN DELINQUENCY STATUS"]
        ]

        time_steps = self._trainX.shape[1]
        feature_names = []
        for t in range(time_steps):
            for name in base_features:
                feature_names.append(f"t{t}_{name}")
        return feature_names

    def _build_wrapper_model(self, trained_model, input_2d_dim, time_steps, feature_dim):
        inputs = Input(shape=(input_2d_dim,))
        reshape_inputs = Reshape((time_steps, feature_dim))(inputs)
        outputs = trained_model(reshape_inputs)
        wrapper = Model(inputs=inputs, outputs=outputs)
        wrapper.trainable = False
        return wrapper

    def explain_models(self, model_list, max_background=200, max_test=200, seed=42):
        try:
            import shap
        except Exception as exc:
            print(f"SHAP is not available. Skipping SHAP explainability. Error: {exc}")
            return

        trainX2D = self._flatten_time_features(self._trainX)
        testX2D = self._flatten_time_features(self._testX)

        background = self._sample_rows(trainX2D, max_background, seed)
        test_sample = self._sample_rows(testX2D, max_test, seed + 1)

        feature_names = self._load_feature_names()

        shap_root = os.path.join(self._data_folder, "shap")
        os.makedirs(shap_root, exist_ok=True)

        for model_instance in model_list:
            trained_model = getattr(model_instance, "_model", None)
            if trained_model is None:
                print(f"Model {getattr(model_instance, 'model_name', 'Unknown')} not trained. Skipping SHAP.")
                continue

            model_name = getattr(model_instance, "model_name", "UnknownModel")
            output_dir = os.path.join(shap_root, model_name)
            os.makedirs(output_dir, exist_ok=True)

            wrapper = self._build_wrapper_model(
                trained_model,
                input_2d_dim=trainX2D.shape[1],
                time_steps=self._trainX.shape[1],
                feature_dim=self._trainX.shape[2],
            )

            explainer = shap.Explainer(wrapper, background)
            try:
                shap_values = explainer.shap_values(test_sample)
            except Exception:
                shap_values = explainer(test_sample)
                if hasattr(shap_values, "values"):
                    shap_values = shap_values.values

            np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
            np.save(os.path.join(output_dir, "test_sample_2d.npy"), test_sample)

            if feature_names is not None:
                feature_names_path = os.path.join(output_dir, "feature_names.json")
                with open(feature_names_path, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, indent=2)

            metadata = {
                "model_name": model_name,
                "train_shape": list(self._trainX.shape),
                "test_shape": list(self._testX.shape),
                "background_rows": int(background.shape[0]),
                "test_rows": int(test_sample.shape[0]),
                "flattened_dim": int(trainX2D.shape[1]),
            }
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            try:
                import matplotlib.pyplot as plt
                if feature_names is not None:
                    shap.summary_plot(
                        shap_values,
                        test_sample,
                        feature_names=feature_names,
                        max_display=200,
                        show=False,
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200)
                    plt.close()

                    explanation = shap.Explanation(
                        values=shap_values,
                        data=test_sample,
                        feature_names=feature_names,
                    )
                    shap.plots.bar(explanation, max_display=100, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=200)
                    plt.close()
            except Exception as exc:
                print(f"SHAP plot generation failed for {model_name}: {exc}")

            print(f"SHAP values saved to {output_dir}")
