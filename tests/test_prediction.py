import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from src.prediction import ADDONS, CATEGORIES, NUMERIC, load_artifacts, validate_customer, predict_customer

@pytest.fixture(scope='module')
def artifacts():
    return load_artifacts()

@pytest.fixture
def customer():
    values = {key: choices[0] for key, choices in CATEGORIES.items()}
    values.update(PhoneService='Yes', MultipleLines='No', tenure=12, MonthlyCharges=70.0, TotalCharges=840.0)
    return values

def test_prediction_and_schema(artifacts, customer):
    model, metadata = artifacts
    frame = validate_customer(customer, metadata['feature_columns'])
    assert list(frame.columns) == list(model.feature_names_in_)
    p, flag = predict_customer(model, metadata, customer)
    assert 0 <= p <= 1
    assert flag == (p >= .5)
    assert p == pytest.approx(model.predict_proba(frame)[0, 1])

@pytest.mark.parametrize('key,value', [('tenure', -1), ('tenure', 73), ('tenure', 1.5), ('MonthlyCharges', float('inf')), ('TotalCharges', float('nan')), ('MonthlyCharges', '70'), ('gender', 'Unknown'), ('ChurnFlag', 1), ('SeniorCitizen', True), ('MultipleLines', 'No phone service'), ('InternetService', 'No')])
def test_invalid_inputs(artifacts, customer, key, value):
    customer[key] = value
    with pytest.raises(ValueError):
        validate_customer(customer, artifacts[1]['feature_columns'])

def test_missing_feature(artifacts, customer):
    del customer['Contract']
    with pytest.raises(ValueError):
        validate_customer(customer, artifacts[1]['feature_columns'])

def test_missing_total_imputed(artifacts, customer):
    customer['TotalCharges'] = None
    assert 0 <= predict_customer(*artifacts, customer)[0] <= 1

@pytest.mark.parametrize('field', list(CATEGORIES) + NUMERIC)
def test_every_field_required(artifacts, customer, field):
    del customer[field]
    with pytest.raises(ValueError):
        validate_customer(customer, artifacts[1]['feature_columns'])

@pytest.mark.parametrize('field', list(CATEGORIES))
def test_every_category_rejects_unknown(artifacts, customer, field):
    customer[field] = 'INVALID'
    with pytest.raises(ValueError):
        validate_customer(customer, artifacts[1]['feature_columns'])

@pytest.mark.parametrize('field,value', [(k, v) for k, choices in CATEGORIES.items() for v in choices])
def test_all_category_options_predict(artifacts, customer, field, value):
    customer[field] = value
    if field == 'PhoneService' and value == 'No':
        customer['MultipleLines'] = 'No phone service'
    if field == 'MultipleLines' and value == 'No phone service':
        customer['PhoneService'] = 'No'
    if (field == 'InternetService' and value == 'No') or value == 'No internet service':
        customer['InternetService'] = 'No'
        customer.update({key: 'No internet service' for key in ADDONS})
    assert 0 <= predict_customer(*artifacts, customer)[0] <= 1

@pytest.mark.parametrize('field', NUMERIC)
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), -float('inf'), True, '1'])
def test_numeric_invalid(artifacts, customer, field, value):
    customer[field] = value
    with pytest.raises(ValueError):
        validate_customer(customer, artifacts[1]['feature_columns'])

@pytest.mark.parametrize('tenure', [0, 72])
def test_tenure_boundaries(artifacts, customer, tenure):
    customer['tenure'] = tenure
    assert 0 <= predict_customer(*artifacts, customer)[0] <= 1

def test_app():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any(m.label == 'Estimated churn probability' for m in app.metric)
    app.number_input[0].set_value(24).run()
    assert not any(m.label == 'Estimated churn probability' for m in app.metric)

def test_all_ui_fields_and_service_dependencies():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
    # Every categorical control is changed, including dynamically constrained services.
    for label in [s.label for s in app.selectbox]:
        widget = next(s for s in app.selectbox if s.label == label)
        options = widget.options
        widget.select(options[-1]).run(timeout=60)
        assert not app.exception
    assert next(s for s in app.selectbox if s.label == 'Internet service').value == 'No'
    for key in ADDONS:
        assert app.selectbox(key=key).value == 'No internet service'
    for i, value in enumerate([72, 110.0, 7900.0]):
        app.number_input[i].set_value(value).run(timeout=60)
        assert not app.exception
    app.checkbox[0].check().run(timeout=60)
    assert app.number_input[2].disabled
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert any(m.label == 'Estimated churn probability' for m in app.metric)
