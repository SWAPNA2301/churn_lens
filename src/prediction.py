"""Validated inference contract for the Phase 2 artifact."""
from pathlib import Path
import json
import math
import hashlib
import sys
import importlib.util
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec('xgboost') is None and (ROOT / '.phase2_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.phase2_packages'))

ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
CATEGORIES = {
    'gender': ['Female', 'Male'], 'SeniorCitizen': [0, 1],
    'Partner': ['No', 'Yes'], 'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'], 'MultipleLines': ['No', 'Yes', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{key: ['No', 'Yes', 'No internet service'] for key in ADDONS},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
}
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']

def load_artifacts():
    metadata = json.loads((ROOT / 'models/model_metadata.json').read_text())
    if metadata['selected_model'] != 'XGBoost':
        raise ValueError('The saved model must be XGBoost.')
    artifact = ROOT / 'models/best_churn_pipeline.joblib'
    expected_hash = (ROOT / 'models/artifact.sha256').read_text().strip()
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Model checksum mismatch. Restore the reviewed artifact.')
    model = joblib.load(artifact)
    if list(model.feature_names_in_) != metadata['feature_columns']:
        raise ValueError('Model and metadata feature schemas differ.')
    if type(model.named_steps['model']).__name__ != 'XGBClassifier':
        raise ValueError('Artifact is not an XGBoost pipeline.')
    if not 0 < metadata['threshold'] < 1:
        raise ValueError('Invalid classification threshold.')
    return model, metadata

def validate_customer(values, columns):
    expected = set(CATEGORIES) | set(NUMERIC)
    if set(columns) != expected or set(values) != expected:
        raise ValueError('Provide exactly the 19 expected customer features.')
    for name, allowed in CATEGORIES.items():
        if values[name] not in allowed or (name == 'SeniorCitizen' and isinstance(values[name], bool)):
            raise ValueError(f'{name}: choose a supported category.')
    cleaned = dict(values)
    for name in NUMERIC:
        value = values[name]
        if name == 'TotalCharges' and value is None:
            cleaned[name] = float('nan')
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name}: enter a finite, non-negative number.')
    if values['tenure'] != int(values['tenure']) or values['tenure'] > 72:
        raise ValueError('tenure: enter whole months between 0 and 72 (training range).')
    if values['PhoneService'] == 'No' and values['MultipleLines'] != 'No phone service':
        raise ValueError('MultipleLines must be No phone service when phone service is absent.')
    if values['PhoneService'] == 'Yes' and values['MultipleLines'] == 'No phone service':
        raise ValueError('Select Yes or No for MultipleLines when phone service is present.')
    for name in ADDONS:
        if (values['InternetService'] == 'No') != (values[name] == 'No internet service'):
            raise ValueError(f'{name}: selection conflicts with internet service.')
    return pd.DataFrame([cleaned], columns=columns)

def predict_customer(model, metadata, values):
    frame = validate_customer(values, metadata['feature_columns'])
    positive = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0, positive])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('The model returned an invalid probability.')
    return probability, probability >= metadata['threshold']
