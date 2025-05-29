import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd

with open('xgboost_model.pkl', 'rb') as file:
   xgboost_model = pickle.load(file)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

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
    
   if st.button("Predict Income"):
      result = predict(age, workclass, final_weight, education, marital_status,
                       occupation, relationship, race, gender,
                       capital_gain, capital_loss, hours_per_week, native_country)
      if result == '>50K':
         st.success("Income more than 50K")
      else:
         st.success("Income less or equal to 50K")



           
def predict(age, workclass, final_weight, education, marital_status, occupation,
            relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country):
 
    input_data = {col: 0 for col in columns}

    input_data['Age'] = float(age)
    input_data['Final Weight'] = float(final_weight)
    input_data['Capital Gain'] = float(capital_gain)
    input_data['Capital Loss'] = float(capital_loss)
    input_data['Hours per Week'] = float(hours_per_week)

    input_data[f'Workclass_{workclass}'] = 1
    input_data[f'Education_{education}'] = 1
    input_data[f'Marital Status_{marital_status}'] = 1
    input_data[f'Occupation_{occupation}'] = 1
    input_data[f'Relationship_{relationship}'] = 1
    input_data[f'Race_{race}'] = 1
    input_data[f'Gender_{gender}'] = 1
    input_data[f'Native Country_{native_country}'] = 1

    df = pd.DataFrame([input_data])

    prediction = xgboost_model.predict(df)[0]
    return '>50K' if prediction == 1 else '<=50K'

if __name__ == "__main__":
    main()
