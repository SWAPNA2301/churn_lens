"""Run with python -m streamlit run app.py."""
import streamlit as st
from src.prediction import ADDONS, CATEGORIES, load_artifacts, predict_customer

st.set_page_config(page_title='ChurnLens | Customer retention', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {max-width:1100px;padding-top:2.5rem}
div[data-testid="stMetric"] {background:#eef4fa;padding:1.2rem;border-radius:12px;color:#142b45}
</style>''', unsafe_allow_html=True)
st.caption('CHURNLENS / CUSTOMER INTELLIGENCE')
st.title('Understand customer churn risk')
st.write('Explore a customer profile using the trained IBM Telco XGBoost pipeline.')

@st.cache_resource
def get_artifacts():
    return load_artifacts()

try:
    model, metadata = get_artifacts()
except Exception:
    st.error('Unable to load the model. Check models/ artifacts and install the versions recorded in model_metadata.json.')
    st.stop()

with st.sidebar:
    st.subheader('Model overview')
    st.write('XGBoost · Phase 2 pipeline')
    st.metric('Test ROC-AUC', f"{metadata['test_metrics']['roc_auc']:.3f}")
    st.caption(f"Classification threshold: {metadata['threshold']:.0%}")
    st.info('This class-weighted model produces an estimated probability, not a calibrated forecast or a guarantee.')
    st.caption('Profiles are used for this calculation only; the app does not save customer inputs.')

values = {}
left, right = st.columns(2, gap='large')
def choose(key, label):
    return st.selectbox(label, CATEGORIES[key], key=key)
with left:
    st.subheader('01 / Customer profile')
    values['gender'] = choose('gender', 'Gender')
    values['SeniorCitizen'] = st.selectbox('Senior citizen', [0, 1], format_func=lambda n: 'Yes' if n else 'No')
    values['Partner'] = choose('Partner', 'Partner')
    values['Dependents'] = choose('Dependents', 'Dependents')
    values['tenure'] = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12, step=1)
    st.subheader('02 / Billing')
    values['Contract'] = choose('Contract', 'Contract')
    values['PaymentMethod'] = choose('PaymentMethod', 'Payment method')
    values['PaperlessBilling'] = choose('PaperlessBilling', 'Paperless billing')
    values['MonthlyCharges'] = st.number_input('Monthly charges ($)', min_value=0.0, value=70.0, step=0.05)
    unknown = st.checkbox('Total charges unavailable', help='The saved pipeline will impute the training-set median.')
    total = st.number_input('Total charges ($)', min_value=0.0, value=840.0, step=0.05, disabled=unknown)
    values['TotalCharges'] = None if unknown else total
with right:
    st.subheader('03 / Services')
    values['PhoneService'] = choose('PhoneService', 'Phone service')
    values['MultipleLines'] = st.selectbox('Multiple lines', ['No', 'Yes'] if values['PhoneService'] == 'Yes' else ['No phone service'])
    values['InternetService'] = choose('InternetService', 'Internet service')
    labels = ['Online security', 'Online backup', 'Device protection', 'Tech support', 'Streaming TV', 'Streaming movies']
    for key, label in zip(ADDONS, labels):
        values[key] = st.selectbox(label, ['No internet service'] if values['InternetService'] == 'No' else ['No', 'Yes'], key=key)
    st.caption('Service-dependent options automatically prevent contradictory profiles.')

if values['MonthlyCharges'] > 118.75 or (values['TotalCharges'] or 0) > 8684.8:
    st.warning('Charges exceed the observed training-data range. Predictions may be less reliable.')
if st.button('Calculate churn risk', type='primary', use_container_width=True):
    try:
        probability, churn = predict_customer(model, metadata, values)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error('Prediction failed. Check the saved model and installed package versions.')
    else:
        st.divider()
        st.subheader('Customer risk assessment')
        a, b = st.columns(2)
        a.metric('Estimated churn probability', f'{probability:.1%}')
        b.metric('Predicted outcome', 'Churn' if churn else 'Stay')
        st.progress(probability)
        st.write('Consider a retention review for this customer.' if churn else 'The estimated risk is below the model threshold. Continue routine customer care.')
        st.caption('Results use the displayed profile and a fixed 50% threshold. Changing any input clears this result until you calculate again.')
