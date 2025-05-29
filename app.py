import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd

# Load model
with open('xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

# HTML Header
html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Income Category Prediction</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
                </div>"""

desc_temp = """ ### Income Category Prediction
This app is used by Credit team 8  
#### Data Source  
Kaggle: Link <Masukkan Link>
"""

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

def run_ml_app():
    st.subheader("Fill in the data to predict income category")
    left, right = st.columns((2,2))

    age = left.number_input('Age', min_value=17, max_value=100)
    workclass = right.selectbox('Workclass', ('Private', 'State-gov', 'Self-emp-not-inc'))
    final_weight = left.number_input('Final Weight')
    education = right.selectbox('Education', (
        'Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad',
        'Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'
    ))
    marital_status = left.selectbox('Marital Status', (
        'Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent'
    ))
    occupation = right.selectbox('Occupation', (
        'Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty',
        'Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving',
        'Priv-house-serv','Protective-serv','Armed-Forces'
    ))
    relationship = left.selectbox('Relationship', (
        'Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried'
    ))
    race = right.selectbox('Race', ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo'))
    gender = left.selectbox('Gender', ('Male', 'Female'))
    capital_gain = right.number_input('Capital Gain')
    capital_loss = left.number_input('Capital Loss')
    hours_per_week = right.number_input('Hours per Week')
    native_country = right.selectbox('Native Country', (
        'United-States','Cambodia','England','Puerto-Rico','Canada','Germany','Outlying-US(Guam-USVI-etc)',
        'India','Japan','Greece','South','China','Cuba','Iran','Honduras','Philippines','Italy','Poland',
        'Jamaica','Vietnam','Mexico','Portugal','Ireland','France','Dominican-Republic','Laos','Ecuador',
        'Taiwan','Haiti','Columbia','Hungary','Guatemala','Nicaragua','Scotland','Thailand','Yugoslavia',
        'El-Salvador','Trinadad&Tobago','Peru','Hong','Holand-Netherlands'
    ))

    if st.button("Predict Income"):
        result = predict(age, workclass, final_weight, education, marital_status, occupation,
                         relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country)

        if result == '>50K':
            st.success(f'Result: Your predicted income is {result}')
        else:
            st.error(f'Result: Your predicted income is {result}')

def predict(age, workclass, final_weight, education, marital_status, occupation,
            relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country):

    # Inisialisasi semua nilai 0
    input_data = {col: 0 for col in columns}

    # Assign fitur numerik
    input_data['age'] = float(age)
    input_data['final_weight'] = float(final_weight)
    input_data['capital_gain'] = float(capital_gain)
    input_data['capital_loss'] = float(capital_loss)
    input_data['hours_per_week'] = float(hours_per_week)

    # One-hot encode categorical input (cek dulu ada kolomnya)
    for cat, val in {
        'workclass': workclass,
        'education': education,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'native_country': native_country
    }.items():
        key = f'{cat}_{val}'
        if key in input_data:
            input_data[key] = 1

    df = pd.DataFrame([input_data])

    try:
        prediction = xgboost_model.predict(df)[0]
        return '>50K' if prediction == 1 else '<=50K'
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 'Prediction Failed'

if __name__ == "__main__":
    main()
