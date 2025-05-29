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
    st.markdown("""<div style="padding:15px;">
                    <h1 style="color:#fff">Income Category Prediction</h1>
                </div""", unsafe_allow_html=True)

    left, right = st.columns((2,2))

    # Inputs
    age = left.number_input('Age', 17, 100)
    final_weight = right.number_input('Final Weight', 0)
    capital_gain = left.number_input('Capital Gain', 0)
    capital_loss = right.number_input('Capital Loss', 0)
    hours_per_week = left.number_input('Hours per Week', 1, 100)
    
    gender = right.selectbox('Gender', ('Male', 'Female'))
    race = left.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    relationship = right.selectbox('Relationship', ['Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife', 'Husband'])
    marital_status = left.selectbox('Marital Status', ['Married-AF-spouse','Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])
    education = right.selectbox('Education', ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'])
    workclass = left.selectbox('Workclass', ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'])
    occupation = right.selectbox('Occupation', ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing','Handlers-cleaners', 'Machine-op-inspct', 'Other-service','Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'])
    native_country = left.selectbox('Native Country', ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba','Dominican-Republic', 'Ecuador', 'El-Salvador', 'England','France', 'Germany', 'Greece', 'Guatemala', 'Haiti','Holand-Netherlands', 'Honduras', 'Hong', 'Hungary','India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan','Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)','Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico','Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago','United-States', 'Vietnam', 'Yugoslavia'])

    if st.button("Predict Income"):
        result = predict(age, final_weight, capital_gain, capital_loss, hours_per_week,
                         gender, race, relationship, marital_status, education, workclass,
                         occupation, native_country)

        if result == '>50K':
            st.success(f'Result: Your predicted income is {result}')
        else:
            st.error(f'Result: Your predicted income is {result}')

def predict(age, final_weight, capital_gain, capital_loss, hours_per_week,
            gender, race, relationship, marital_status, education, workclass,
            occupation, native_country):

    input_data = {
        'Age': age,
        'Final Weight': final_weight,
        'Capital Gain': capital_gain,
        'capital loss': capital_loss,
        'Hours per Week': hours_per_week,
        f'Gender_ {gender}': 1,
        f'Race_ {race}': 1,
        f'Relationship_ {relationship}': 1,
        f'Marital Status_ {marital_status}': 1,
        f'EducationNum': 0,  # Default jika diperlukan
        f'Workclass_ {workclass}': 1,
        f'Occupation_ {occupation}': 1,
        f'Native Country_ {native_country}': 1
    }

    # Lengkapi kolom kosong supaya sesuai dengan model
    all_cols = xgboost_model.get_booster().feature_names
    for col in all_cols:
        if col not in input_data:
            input_data[col] = 0

    df = pd.DataFrame([input_data])
    prediction = xgboost_model.predict(df)[0]
    return '>50K' if prediction == 1 else '<=50K'

if __name__ == "__main__":
    main()
