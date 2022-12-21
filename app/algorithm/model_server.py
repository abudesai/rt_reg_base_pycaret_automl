import os
import sys

import algorithm.model.classifier as classifier
import algorithm.utils as utils
import numpy as np, pandas as pd

# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "regressionBaseMainInput"
        ]["idField"]
        self.model = None

    def _get_model(self):
        self.model = classifier.load_model(self.model_path)
        return self.model

    def predict(self, data):
        # preprocessor = self._get_preprocessor()
        model = self._get_model()
        if model is None:
            raise Exception("No model found. Did you train first?")

        # make predictions
        preds = model.predict(data)

        # return the prediction df with the id and prediction fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = preds.values

        return preds_df

    def predict_to_json(self, data):
        preds_df = self.predict(data)

        predictions_response = []
        for rec in preds_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[self.id_field_name] = rec[self.id_field_name]
            pred_obj["prediction"] = np.round(rec["prediction"], 5)
            predictions_response.append(pred_obj)
        return predictions_response
