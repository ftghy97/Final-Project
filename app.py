import streamlit as st
import streamlit.components.v1 as stc
import pickle

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
    

    #If button is clilcked
    pass

def predict(Age, Workclass, Final_Weight, EducationNum, Marital_Status, Occupation, Relationship, Race, Gender, Capital_Gain, Capital_loss, Hours_per_week, Native_Country, Income):
    
    #Making prediction
    pass

if __name__ == "__main__":
    main()
