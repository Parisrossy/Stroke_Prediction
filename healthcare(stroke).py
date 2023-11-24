import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE


data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data.head()

df = data.copy()

# clean data
def cleaner(dataframe):
    for i in dataframe.columns:
        if (dataframe[i].isnull().sum()/len(dataframe) * 100) > 30:
            dataframe.drop(i, inplace = True, axis = 1)

        elif dataframe[i].dtypes != 'O':
            dataframe[i].fillna(dataframe[i].median(), inplace = True)

        else:
            dataframe[i].fillna(dataframe[i].mode()[0], inplace = True)

    (dataframe.isnull().sum().sort_values(ascending = False).head())
    return dataframe

cleaner(df)

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

df.drop('id', axis = 1, inplace = True)

# Transform data
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: 
    if i in df.drop('stroke', axis = 1).columns: 
        df[i] = scaler.fit_transform(df[[i]]) 
for i in categoricals.columns:
    if i in df.drop('stroke', axis = 1).columns: 
        df[i] = encoder.fit_transform(df[i])

y = df['stroke']
x = df.drop('stroke',axis = 1)

# # - Using XGBOOST to find feature importance
# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x, y)

# # Print feature importance scores
# xgb.plot_importance(model)

#----------Feature Selection--------
sel_cols = ['avg_glucose_level', 'bmi', 'age', 'smoking_status', 'Residence_type', 'gender', 'work_type']
x = df[sel_cols]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 75, stratify = y)

# Modelling
model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# %pip install imbalanced-learn
# from imblearn.over_sampling import SMOTE
# import pandas as pd
# import seaborn as sns

# Assuming you have a Datadf df with features (x) and target variable (y)
# Replace 'Diabetes' with the actual name of your target variable

# Extract features (x) and target variable (y)
y = df.stroke
x = df.drop('stroke', axis=1)

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto',  random_state=42)  # You can adjust the sampling_strategy as needed

# Apply SMOTE to generate synthetic samples
x_resampled, y_resampled = smote.fit_resample(x, y)

# Create a new Datadf with the resampled data
ds = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_resampled, name='stroke')], axis=1)

# Plot the count of samples for each class in the resampled data
sns.countplot(x=ds['stroke'])
model = pickle.dump(model, open('healthcare(stroke).pkl', 'wb'))
print('\nModel is saved\n')

#-------Streamlit development------
model = pickle.load(open('healthcare(stroke).pkl', "rb"))

st.markdown("<h1 style = 'color: #B31312; text-align: center;font-family: Arial, Helvetica, sans-serif; '>STROKE PREDICTION</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #2B2A4C; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BY OLUWAYOMI ROSEMARY</h3>", unsafe_allow_html= True)
st.image('pngwing.com (19).png', width = 400)
st.markdown("<h2 style = 'color: #2B2A4C; background: #B31312; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown('<br>', unsafe_allow_html= True)
st.markdown("<p>Stroke prediction involves assessing an individual's risk of experiencing a stroke based on various factors. This predictive analysis aims to identify those at higher risk to initiate preventive measures.</p>",unsafe_allow_html= True)

password = ['one', 'two', 'three', 'four']
username = st.text_input('Pls enter your username')
passes = st.text_input('Pls input password')

if passes in password:
    st.toast('Registered User')
    print(f'Welcome {username}, Pls enjoy your usage as a registered user')
#else:
    #st.error('You are not a registered user. But you have three trials')

    st.sidebar.image('pngwing.com3.png', caption = f'Welcome {username}')

# dx = data[['avg_glucose_level', 'bmi', 'age', 'smoking_status', 'Residence_type', 'gender', 'work_type']]
# st.write(data.head(3))

    avg_glucose = st.sidebar.slider("Average glucose level in blood", data['avg_glucose_level'].min(), data['avg_glucose_level'].max())
    bmi = st.sidebar.slider("Body Mass Index", data['bmi'].min(), data['bmi'].max())
    age = st.sidebar.slider("Age", data['age'].min(), data['age'].max())
    smoking_status = st.sidebar.selectbox("Smoking status of the patient", data['smoking_status'].unique())
    Residence_type = st.sidebar.selectbox("Residence type of the patient", data['Residence_type'].unique())
    gender = st.sidebar.selectbox("Gender", data['gender'].unique())
    work_type = st.sidebar.selectbox("Work type of the patient", data['work_type'].unique())

    st.header('Input Data')

    # Bring all the inputs into a dataframe
    input_data = pd.Series({
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'age': age,
        'smoking_status': smoking_status,
        'Residence_type': Residence_type,
        'gender': gender,
        'work_type': work_type
    })

    # Reshape the Series to a DataFrame
    input_variable = input_data.to_frame().T

    st.write(input_variable)


    categorical = input_variable.select_dtypes(include = ['object', 'category'])
    numerical = input_variable.select_dtypes(include = 'number')


    # Standard Scale the Input Variable.
    for i in numerical.columns:
        if i in input_variable.columns:
            input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])
    for i in categorical.columns:
        if i in input_variable.columns: 
            input_variable[i] = LabelEncoder().fit_transform(input_variable[i])

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

    if st.button('Press To Predict'):
        predicted = model.predict(input_variable)
        st.toast('Stroke Status Predicted')
        st.image('pngwing.com2.png', width = 100)
        st.success(f'{predicted} predicted')
        if predicted == 0:
            st.success('The patient does not have stroke')
        else:
            st.success('The patient has a stroke')

else:
    st.error('You are not a registered user. But you have three trials')
