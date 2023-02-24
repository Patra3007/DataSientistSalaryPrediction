import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as t

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from sklearn.model_selection import KFold, cross_val_score
from sklearn import ensemble

def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')

header = st.container()

model_type = st.container()

features = st.container()

modelTraining = st.container()

Neural = False
y_pred = 0


with header:
    st.title('Data Sientist Salary Prediction Site')
    
    #input for streamlit
with features:
    st.header('Input your information')
    exp = st.selectbox('Chose your experience level', options=[ 'Entry-level','Mid-level','Senior-level','Executive-level'],index=0)
    time = st.selectbox('Chose your employment type', options=[ 'Full time','Part time','Flexible'],index=0)
    job_title = st.selectbox('Chose your job title', options=[ 'Data Scientist' ,'Machine Learning Scientist' ,'Big Data Engineer',
     'Product Data Analyst', 'Machine Learning Engineer' ,'Data Analyst',
     'Lead Data Scientist' ,'Business Data Analyst' ,'Lead Data Engineer',
     'Lead Data Analyst' ,'Data Engineer', 'Data Science Consultant',
     'BI Data Analyst' ,'Director of Data Science', 'Research Scientist',
     'Machine Learning Manager', 'Data Engineering Manager',
     'Machine Learning Infrastructure Engineer', 'ML Engineer', 'AI Scientist',
     'Computer Vision Engineer' ,'Principal Data Scientist',
     'Data Science Manager' ,'Head of Data' ,'3D Computer Vision Researcher',
     'Data Analytics Engineer' ,'Applied Data Scientist',
     'Marketing Data Analyst' ,'Cloud Data Engineer', 'Financial Data Analyst',
     'Computer Vision Software Engineer', 'Director of Data Engineering',
     'Data Science Engineer', 'Principal Data Engineer',
     'Machine Learning Developer' ,'Applied Machine Learning Scientist',
     'Data Analytics Manager' ,'Head of Data Science', 'Data Specialist',
     'Data Architect' ,'Finance Data Analyst', 'Principal Data Analyst',
     'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer',
     'ETL Developer' ,'Head of Machine Learning', 'NLP Engineer',
     'Lead Machine Learning Engineer', 'Data Analytics Lead'],index=0)
    country = st.selectbox('Chose your residence', options=['DE' ,'JP' ,'GB' ,'HN', 'US' ,'HU', 'NZ' ,'FR','IN', 'PK' ,'PL' ,'PT' ,'CN', 'GR',
     'AE', 'NL', 'MX' ,'CA', 'AT', 'NG' ,'PH' ,'ES' ,'DK' ,'RU' ,'IT', 'HR' ,'BG', 'SG',
     'BR' ,'IQ', 'VN', 'BE' ,'UA', 'MT' ,'CL', 'RO' ,'IR' ,'CO', 'MD', 'KE', 'SI', 'HK',
     'TR', 'RS', 'PR', 'LU' ,'JE', 'CZ', 'AR' ,'DZ' ,'TN' ,'MY', 'EE', 'AU', 'BO' ,'IE',
     'CH'],index=0)
    remote = st.selectbox('Chose your remote ratio', options=[ 0 , 50 ,100,],index=0)
    country_company = st.selectbox('Chose your company residence', options=['DE' ,'JP', 'GB' ,'HN' ,'US' ,'HU', 'NZ' ,'FR' ,'IN', 'PK', 'CN', 'GR' ,'AE', 'NL',
     'MX' ,'CA','AT' ,'NG', 'ES' ,'PT', 'DK' ,'IT', 'HR' ,'LU', 'PL', 'SG' ,'RO', 'IQ',
     'BR' ,'BE' ,'UA', 'IL', 'RU' ,'MT' ,'CL' ,'IR' ,'CO', 'MD', 'KE' ,'SI', 'CH' ,'VN',
     'AS' ,'TR' ,'CZ' ,'DZ' ,'EE', 'MY' ,'AU' ,'IE'],index=0)
    company_size = st.selectbox('Chose your company size', options=['L' ,'M','S'],index=0)
    
    #chosing of the model in streamlit
with model_type:
    st.header('Chose algoritham for the prediction')
    mod = st.radio(' ',options=['KNN', 'RandomForest','NeuralNetwork'])
    
with modelTraining:
    clicked = st.button("Calculate")
    
        #data preparation for traning models
    data = pd.read_csv("ds_salaries.csv")

    X = data.drop(['Unnamed: 0','salary', 'salary_currency','salary_in_usd'], axis=1)
    X213 = X
    y = data['salary_in_usd']
    
             
    if(exp == 'Entry-level'):
       exp = 'EN'
 
    elif(exp == 'Mid-level'):
        exp = 'MI'

    elif(exp == 'Senior-level'):
       exp = 'SE'

    else:
       exp = 'EX'

       
    if(time == 'Full time'):
       time = 'FT'

    elif(time == 'Part time'):
       time = 'PT'

    else:
       exp = 'FL'
   

    user_input = {'work_year': [2022],
             'experience_level': [exp],
             'employment_type': [time],
             'job_title' :[job_title],
             'employee_residence':[country],
             'remote_ratio':[remote],
             'company_location':[country_company],
             'company_size':[company_size]}
    
    input_data = pd.DataFrame(user_input)
    
    X = X.append(input_data,ignore_index=True)
    
    X = pd.get_dummies(X)
    
    input_data = X.tail(1)
    
    X.drop(X.tail(1).index,inplace=True)


    #split to 70%train 15%test and 15%val
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, random_state=1)

    X_test, X_valid, y_test, y_valid = train_test_split(X_rest, y_rest, test_size=0.50, random_state=1)

    
    
   
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)
    input_data = scaler.transform(input_data)

    #calculating the perfect number of estimators and number of neigbours for algorithams
    # n_estimators = [1, 2, 3, 4 ,5 ,6 ,7 ,8 ,9 ,10, 20, 30]
    # for estimator in n_estimators:
    #     score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= estimator, random_state= 42), X_valid, y_valid, scoring="neg_mean_squared_error")
    #     print(f'{estimator}')
    #     rmse(score.mean())
        
    # n_neigbours =  [2,3,4,5,6,7,8,9,10]
    # for neigbour in n_neigbours:
    #     score = cross_val_score(KNeighborsRegressor(n_neighbors= neigbour), X_valid, y_valid, scoring="neg_mean_squared_error")
    #     print(f'{neigbour}')
    #     rmse(score.mean())
        
    
    if clicked:
        if(mod == 'KNN'):   
            model = KNeighborsRegressor(n_neighbors=5)
            Neural = False
        elif(mod == 'RandomForest'):
            model = RandomForestRegressor(criterion= "squared_error",n_estimators=8,random_state=42)
            Neural = False
        else:    
            Neural = True 

    
 
   
        if(Neural):
            #asembling of the simple neural network
            st.write('Calculating, please wait!')
            NeuralNetwork = keras.Sequential();
            NeuralNetwork.add(keras.layers.Dense(50, input_dim=170,activation='relu'))
            NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
            NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
            NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
            NeuralNetwork.add(keras.layers.Dense(1,activation='linear'))
            NeuralNetwork.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

            history = NeuralNetwork.fit(X_train,y_train, validation_split=0.2,epochs=50,batch_size= 10)
            
            #predicting with user input
            y_pred = NeuralNetwork.predict(input_data)
            Neural = False
        else:
                #predicting with user input
                model.fit(X_train,y_train)
                y_pred = model.predict(input_data)
        
 
        
       #printing the prediction
        st.spinner(text="Calculating...")
        rounded =int(y_pred[0])
        value = str(rounded)
        st.write('Predicted annual salary : ' + value + ' $')

models = {
    "KNN":KNeighborsRegressor(n_neighbors=5),
    "RandomForest":RandomForestRegressor(criterion= "squared_error",n_estimators=8,random_state=42)
}


#model plot 
# for name , model in models.items():
    
#     model.fit(X_train,y_train)
    
#     y_pred = model.predict(X_test)
         
#     X_pos = np.arange(start=0,stop=91,step=1)
#     plt.plot(X_pos,y_test)
#     plt.plot(X_pos,y_pred)
#     plt.xlabel('y test')
#     plt.legend(['y_test','y_pred'])
#     plt.ylabel('y pred')
#     plt.title(name)
#     plt.show()
   
# NeuralNetwork = keras.Sequential()
# NeuralNetwork.add(keras.layers.Dense(50, input_dim=170,activation='relu'))
# NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
# NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
# NeuralNetwork.add(keras.layers.Dense(50,activation='relu'))
# NeuralNetwork.add(keras.layers.Dense(1,activation='linear'))
# NeuralNetwork.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

# history = NeuralNetwork.fit(X_train,y_train, validation_split=0.2,epochs=50,batch_size= 10)
   
   
# y_pred = NeuralNetwork.predict(X_test)
# X_pos = np.arange(start=0,stop=91,step=1)
# plt.plot(X_pos,y_test)
# plt.plot(X_pos,y_pred)
   
#  # naming the x axis
# plt.xlabel(' y test')
#  # naming the y axis
# plt.ylabel(' y pred')
# plt.legend(['y_test','y_pred'])
# plt.title("Neural")
#  # function to show the plot
# plt.show()
