import pandas as pd
import joblib
from pathlib import Path

class SupplierRiskPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)

        # Load trained artifacts
        self.model = joblib.load(self.model_dir / "stacking_model.pkl")
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        self.freq_mappings = joblib.load(self.model_dir / "freq_mappings.pkl")
        self.feature_cols = joblib.load(self.model_dir / "feature_columns.pkl")
        self.median_values = joblib.load(self.model_dir / "median_values.pkl")

        if hasattr(self.scaler, "feature_names_in_"):
            self.numeric_cols = list(self.scaler.feature_names_in_)
        else:
            self.numeric_cols = [
                col for col in self.feature_cols if not col.endswith("_freq")
            ]

    def preprocess(self, raw_data: dict) -> pd.DataFrame:
        """
        Preprocess raw supplier data into the format required by the model.
        Includes fixes for data type consistency between JSON and Pandas.
        """
        df = pd.DataFrame([raw_data])

        # Robust Frequency Encoding
        for col, mapping in self.freq_mappings.items():
            if col in df.columns and mapping:
                # Detect the type used during training (e.g., int or str)
                sample_key = list(mapping.keys())[0]
                target_type = type(sample_key)
                
                # Convert input to that type before mapping to avoid NaN results
                df[col + "_freq"] = df[col].astype(target_type).map(mapping).fillna(0)
            else:
                df[col + "_freq"] = 0

        # 2. Drop original categorical columns
        df = df.drop(columns=self.freq_mappings.keys(), errors="ignore")

        # 3. Ensure all trained features exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # 4. Strict feature order
        df = df[self.feature_cols]

        # 5. Handle Numeric Conversion and Scaling
        numeric_cols_present = [col for col in self.numeric_cols if col in df.columns]
        for col in numeric_cols_present:
            # Ensure values are numeric (FastAPI might send strings)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(self.median_values.get(col, 0))

        if numeric_cols_present:
            df[numeric_cols_present] = self.scaler.transform(df[numeric_cols_present])

        return df

    def predict(self, raw_data: dict) -> dict:
        """
        Predict risk for a single supplier.
        Returns 'risk_score' and 'is_high_risk' (0 or 1).
        """
        df_final = self.preprocess(raw_data)

        # Get probability and final classification
        probability = self.model.predict_proba(df_final)[0][1]
        is_high_risk = self.model.predict(df_final)[0]

        return {
            "risk_score": round(float(probability), 4),
            "is_high_risk": int(is_high_risk),
        }