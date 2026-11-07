"""Independently reproduce holdout metrics without fitting or replacing models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.prediction import ROOT, load_artifacts, predict_customer

def test_saved_evaluation_and_preprocessing():
    model, meta = load_artifacts()
    data = pd.read_csv(ROOT / 'data/raw/Telco-Customer-Churn.txt', na_values=['', ' '])
    for col in data.select_dtypes(include='object'):
        data[col] = data[col].str.strip()
    data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
    data = data.drop_duplicates()
    assert data.customerID.is_unique
    X = data.drop(columns=['customerID', 'Churn'])
    y = data.Churn.map({'No': 0, 'Yes': 1})
    train, test, yt, ytest = train_test_split(X, y, test_size=.2, stratify=y, random_state=meta['seed'])
    assert len(train) == meta['train_rows'] and len(test) == meta['test_rows']
    assert set(train.index).isdisjoint(test.index)
    assert not {'Churn', 'ChurnFlag', 'customerID'}.intersection(model.feature_names_in_)
    probs = model.predict_proba(test)[:, list(model.classes_).index(1)]
    pred = probs >= meta['threshold']
    for name, function in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
        assert function(ytest, pred) == pytest.approx(meta['test_metrics'][name], abs=1e-10)
    assert roc_auc_score(ytest, probs) == pytest.approx(meta['test_metrics']['roc_auc'], abs=1e-10)
    assert confusion_matrix(ytest, pred).tolist() == [[755, 280], [77, 297]]
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    prep = model.named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(prep.named_steps['impute'].statistics_, train[numeric].median())
    filled = train[numeric].fillna(train[numeric].median())
    np.testing.assert_allclose(prep.named_steps['scale'].mean_, filled.mean())
    np.testing.assert_allclose(prep.named_steps['scale'].var_, filled.var(ddof=0))
    assert prep.named_steps['scale'].n_samples_seen_ == len(train)
    # Check real held-out customers through the same validation path as the app.
    for index in [int(np.argmin(probs)), int(np.argmax(probs)), 0]:
        customer = test.iloc[index].to_dict()
        if pd.isna(customer['TotalCharges']):
            customer['TotalCharges'] = None
        assert predict_customer(model, meta, customer)[0] == pytest.approx(probs[index])
    scores = pd.read_csv(ROOT / 'reports/phase2_cv_scores.csv')
    assert scores.groupby('model').size().eq(5).all()
    assert not scores.duplicated(['model', 'fold']).any()
    means = scores.groupby('model').roc_auc.mean()
    assert means.idxmax() == meta['selected_model']
    assert means.max() == pytest.approx(meta['cv_roc_auc'])
    print('\nVerified test metrics:', meta['test_metrics'])
    print('Real held-out probability range:', float(probs.min()), float(probs.max()))
