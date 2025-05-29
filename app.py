import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd

with open('xgboost_model.pkl', 'rb') as file:
   xgboost_model = pickle.load(file)


html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Income Category Prediction</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
                """

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
    design = """<div style="padding:15px;">
                    <h1 style="color:#fff">Income Category Prediction</h1>
                </div
             """
    st.markdown(design, unsafe_allow_html=True)
    #Membuat Struktur Form
    left, right = st.columns((2,2))
    age = left.number_input(label = 'Age',
                            min_value = 17, max_value = 100)
    workclass = right.selectbox('Workclass', ('Private', 'State-gov', 'Self-emp-not-inc'))
    final_weight = left.text_input('Final Weight')
    education = right.selectbox('Education', (
    'Preschool',
    '1st-4th',
    '5th-6th',
    '7th-8th',
    '9th',
    '10th',
    '11th',
    '12th',
    'HS-grad',
    'Some-college',
    'Assoc-voc',
    'Assoc-acdm',
    'Bachelors',
    'Masters',
    'Prof-school',
    'Doctorate'
))
    
    marital_status = left.selectbox('Marital Status', (
    'Married-civ-spouse',
    'Divorced',
    'Never-married',
    'Separated',
    'Widowed',
    'Married-spouse-absent'
))
    occupation = right.selectbox('Occupation', (
    'Tech-support',
    'Craft-repair',
    'Other-service',
    'Sales',
    'Exec-managerial',
    'Prof-specialty',
    'Handlers-cleaners',
    'Machine-op-inspct',
    'Adm-clerical',
    'Farming-fishing',
    'Transport-moving',
    'Priv-house-serv',
    'Protective-serv',
    'Armed-Forces'
))
    relationship = left.selectbox('Relationship', (
    'Wife',
    'Own-child',
    'Husband',
    'Not-in-family',
    'Other-relative',
    'Unmarried'
))
    race = right.selectbox(
        'Race',
        ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo')
    )


    gender = left.selectbox('Gender', ('Male', 'Female'))
    capital_gain = right.text_input('Capital_Gain')
    capital_loss = left.text_input('Capital_loss')
    hours_per_week = right.text_input('Hours_per_week')
    native_country = left.selectbox('Native Country',('United-States','Cambodia','England','Puerto-Rico','Canada','Germany','Outlying-US(Guam-USVI-etc)',
                                    'India', 'Japan','Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras','Philippines', 'Italy','Poland','Jamaica', 'Vietnam', 
                                    'Mexico','Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos','Ecuador','Taiwan', 'Haiti','Columbia', 'Hungary',
                                    'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong','Holand-Netherlands'))
    
    
    
    #If button is clilcked
    if st.button("Predict Income"):
        result = predict(age, workclass, final_weight, education, marital_status, occupation,
            relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country)

        if result == '>50k':
            st.success(f'Result: Your predicted income is {result}')
        else:
            st.error(f'Result: Your predicted income is {result} ')

def predict(age, workclass, final_weight, education, marital_status, occupation,
            relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country):

       columns = [
        'age', 'final_weight', 'capital_gain', 'capital_loss', 'hours_per_week',
        'workclass_Private', 'workclass_State-gov', 'workclass_Self-emp-not-inc',
        'education_Preschool', 'education_1st-4th', 'education_5th-6th', 'education_7th-8th',
        'education_9th', 'education_10th', 'education_11th', 'education_12th', 'education_HS-grad',
        'education_Some-college', 'education_Assoc-voc', 'education_Assoc-acdm', 'education_Bachelors',
        'education_Masters', 'education_Prof-school', 'education_Doctorate',
        'marital_status_Married-civ-spouse', 'marital_status_Divorced', 'marital_status_Never-married',
        'marital_status_Separated', 'marital_status_Widowed', 'marital_status_Married-spouse-absent',
        'occupation_Tech-support', 'occupation_Craft-repair', 'occupation_Other-service', 
        'occupation_Sales', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
        'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Adm-clerical',
        'occupation_Farming-fishing', 'occupation_Transport-moving', 'occupation_Priv-house-serv',
        'occupation_Protective-serv', 'occupation_Armed-Forces',
        'relationship_Wife', 'relationship_Own-child', 'relationship_Husband', 'relationship_Not-in-family',
        'relationship_Other-relative', 'relationship_Unmarried',
        'race_White', 'race_Black', 'race_Asian-Pac-Islander', 'race_Amer-Indian-Eskimo',
        'gender_Male', 'gender_Female',
        'native_country_United-States', 'native_country_Cambodia', 'native_country_England', 
        'native_country_Puerto-Rico', 'native_country_Canada', 'native_country_Germany', 
        'native_country_Outlying-US(Guam-USVI-etc)', 'native_country_India', 'native_country_Japan',
        'native_country_Greece', 'native_country_South', 'native_country_China', 'native_country_Cuba',
        'native_country_Iran', 'native_country_Honduras', 'native_country_Philippines',
        'native_country_Italy', 'native_country_Poland', 'native_country_Jamaica',
        'native_country_Vietnam', 'native_country_Mexico', 'native_country_Portugal',
        'native_country_Ireland', 'native_country_France', 'native_country_Dominican-Republic',
        'native_country_Laos', 'native_country_Ecuador', 'native_country_Taiwan',
        'native_country_Haiti', 'native_country_Columbia', 'native_country_Hungary',
        'native_country_Guatemala', 'native_country_Nicaragua', 'native_country_Scotland',
        'native_country_Thailand', 'native_country_Yugoslavia', 'native_country_El-Salvador',
        'native_country_Trinadad&Tobago', 'native_country_Peru', 'native_country_Hong',
        'native_country_Holand-Netherlands'
    ]


    

    

    # Buat DataFrame
    df = pd.DataFrame([input_data])

    # Prediksi
    prediction = xgboost_model.predict(df)[0]
    return '>50K' if prediction == 1 else '<=50K'

if __name__ == "__main__":
    main()
