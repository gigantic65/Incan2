import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random
import itertools
from tensorflow.keras.models import load_model 

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    

def app():
    
    st.write('')
    st.write('')
    
    st.markdown("<h6 style='text-align: right; color: black;'>적용 제품: Incan UT6581, UT578A, UT578AF, UT578AS제품 </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: right; color: black;'>총 데이터 갯수: 3740 Cases (Base A, B, C 조건)</h6>", unsafe_allow_html=True)

    st.write("")

    
    with st.expander("Predict New Conditions Guide"):
        st.write(
                "1. 정확도 확인 : Model accuracy 버튼 클릭.\n"
                "2. 조색제 투입량에 따른 색차 예측과 초기 색차에 따른 조색제 량 예측.\n"
        )


    st.sidebar.header('1. 조색 예측모델 업로드 ')
    st.sidebar.write("<h5 style='text-align: left; color: black;'> - ▷ 머신러닝 모델 업로드 완료 </h5>", unsafe_allow_html=True)
    #st.sidebar.write('자동 업로드')
    
    
    df1 = pd.read_csv('train_incan_11.csv')
    models1 = load_model('model_11.h5')
    scaler1 = pickle.load(open('./scaler_11.pkl', 'rb'))
    
    df2 = pd.read_csv('train_incan_22.csv')
    models2 = load_model('model_22.h5')
    scaler2 = pickle.load(open('./scaler_22.pkl', 'rb'))
    
    df3 = pd.read_csv('train_incan_33.csv')
    models3 = load_model('model_33.h5')
    scaler3 = pickle.load(open('./scaler_33.pkl', 'rb'))
    
    df4 = pd.read_csv('train_incan_44.csv')
    models4 = load_model('model_44.h5')
    scaler4 = pickle.load(open('./scaler_44.pkl', 'rb'))
    
    df5 = pd.read_csv('train_incan_55.csv')
    models5 = load_model('model_55.h5')
    scaler5 = pickle.load(open('./scaler_55.pkl', 'rb'))
    
    df6 = pd.read_csv('train_incan_66.csv')
    models6 = load_model('model_66.h5')
    scaler6 = pickle.load(open('./scaler_66.pkl', 'rb'))
    
    
    df7 = pd.read_csv('train_incan_77.csv')
    models7 = load_model('model_77.h5')
    scaler7 = pickle.load(open('./scaler_77.pkl', 'rb'))
    



    st.sidebar.write('')

    st.subheader('1. 조색 예측 모델 정보')
    st.write('')

    df = df3
    models = models3
    scaler = scaler3
        
    x = list(df.columns[:-3])
    x2 = list(df.columns[:-10])
    y = list(df.columns[df.shape[1]-3:])

        #Selected_X = st.sidebar.multiselect('X variables', x, x)
        #Selected_y = st.sidebar.multiselect('Y variables', y, y)
            
    Selected_X = np.array(x)
    Selected_X2 = np.array(x2)
    Selected_y = np.array(y)
        
    st.write('**1.1 X인자 수:**',Selected_X2.shape[0],'**학습된 조색제 수:**',Selected_X2.shape[0])
    st.info(list(Selected_X2))

    
    st.write('**1.2 Y인자 수:**',Selected_y.shape[0])
    st.info(list(Selected_y))



   
            
    st.write('')   
            
    st.write('**1.3 선정 예측 모델 :**')
    st.info('Deep_Learning Model')
    st.write('')
            
        
    st.write('**1.4 예측 모델 정확도 :**')
    

 
    R2 = []

    columns = []    

    model = models

   
    X = df[Selected_X]
    y = df[Selected_y]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    rescaledTestX = scaler.transform(X_test)
        
    model = models
     

            
    if st.button('Check Model Accuracy'):
        

        predictions = model.predict(rescaledTestX)


        y_test = np.array(y_test)
        A_column = ['Actual_L','Actual_a','Actual_b']
        y_test = pd.DataFrame(y_test,columns=A_column)
        
        
        P_column = ['Pred_L','Pred_a','Pred_b']
        predictions = pd.DataFrame(predictions,columns=P_column)
        
        
        new_result = pd.concat((y_test, predictions), axis=1)
        new_result['Delta_E'] = ((y_test['Actual_L'] - predictions['Pred_L'])**2+(y_test['Actual_a'] - predictions['Pred_a'])**2+(y_test['Actual_b'] - predictions['Pred_b'])**2)**0.5 
        
    
        R2.append('%f' %  round(sm.r2_score(y_test,predictions),5))

        
        st.write('Model Accuracy for Test data ($R^2$):')
            
        st.info( R2[0] )
        
        st.write(new_result)
                
                
        length = range(y_test.shape[0])
                
            #fig, axs = plt.subplots(ncols=3)
            
        fig, ax = plt.subplots(figsize=(10,4))
            
        g = sns.lineplot(x=length,y=predictions['Pred_L'],color='blue',label='prediction')
        g = sns.lineplot(x=length,y=y_test['Actual_L'],ax=g.axes, color='red',label='actual')
        g.set_ylabel("L", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            
        g1 = sns.lineplot(x=length,y=predictions['Pred_a'],color='blue',label='prediction')
        g1 = sns.lineplot(x=length,y=y_test['Actual_a'],ax=g1.axes, color='red',label='actual')
        g1.set_ylabel("a", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            
        g2 = sns.lineplot(x=length,y=predictions['Pred_b'],color='blue',label='prediction')
        g2 = sns.lineplot(x=length,y=y_test['Actual_b'],ax=g2.axes, color='red',label='actual')
        g2.set_ylabel("b", fontsize = 10)
        plt.legend()
            
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
            
        st.pyplot()
                
            
 
    #st.sidebar.write('3.1 Predict Single Condition')
            
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')


    
    st.sidebar.header('2. 조색 배합 예측')
    
    st.sidebar.write("<h5 style='text-align: left; color: black;'> 2.1 기존 유사 조색 배합 확인</h5>", unsafe_allow_html=True)
    st.sidebar.write('')
    st.sidebar.write("<h5 style='text-align: left; color: black;'> 2.2 신규 조색 배합 설계 </h5>", unsafe_allow_html=True)
    st.sidebar.write("<h5 style='text-align: left; color: black;'> -      ▷ 기존배합 수정설계 </h5>", unsafe_allow_html=True)
    st.sidebar.write("<h5 style='text-align: left; color: black;'> -      ▷ 신규배합 자동설계 </h5>", unsafe_allow_html=True)
    st.sidebar.write("<h5 style='text-align: left; color: black;'> -      ▷신규배합 수동설계 </h5>", unsafe_allow_html=True)
    

    
    #st.sidebar.write('**2.1 기존 유사 조색 배합 예측**')
    #select = [1.5,0.5,1.0,2.0,5.0]
    #number = st.sidebar.selectbox("기존 유사배합 색차기준 : ", select)
                            
    

    st.write('')
    st.write('')
    st.write('')        
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
                             
    st.subheader('**2. 조색 배합 예측**')
                
    Target_n = []
    Target_v = []
        
            
    st.write('**2.1 기존 유사 조색 배합 확인**')


    col1,col2 = st.columns([1,1])
    
    with col1:
        select = ['Select','UT578_A','UT578_AS','UT578_AF','UT6581']
        selected1 = st.selectbox("제품 선정 : ", select)
        
        
    with col2:
        select = ['Select','Base_A','Base_B','Base_C']
        selected2 = st.selectbox("배합 베이스 선정 : ", select)



        
    """X = df[Selected_X]
    y = df[Selected_y]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)"""
        
    
        
    #st.write('Target_Color')
    
    columns = ['Target_L','Target_a','Target_b']
        

    Target_n = []
    Target_v = []
        
    col1,col2,col3 = st.columns([1,1,1])
        
    with col1:
        value1 = st.number_input(columns[0], -1000.00, 1000.00, 0.0,format="%.3f")
        Target_n.append(columns[0])
        Target_v.append(value1)
    with col2:
        value2 = st.number_input(columns[1], -1000.00, 1000.00, 0.0,format="%.3f")
        Target_n.append(columns[1])
        Target_v.append(value2)
    with col3:
        value3 = st.number_input(columns[2], -1000.00, 1000.00, 0.0,format="%.3f")
        Target_n.append(columns[2])
        Target_v.append(value3)
            

        
    name2=[]
    test2=[]

    
    
    color_list =[]   
    color_list = ['SK1','SK2','SB1','SB2','SG1','SY1','SY2','SY3','SO1','SP1','SV1','SR1','SR2','SR3']
    color_list = pd.DataFrame(color_list,columns=['color'])
    
    DT = pd.read_csv('incan_data_r0.csv')

    count = 0
    
    para3 = pd.DataFrame()
    
    
    if st.button('Run Prediction',key = count):
        
                  
        #DT = pd.read_csv('incan_data_r0.csv')

        
        if selected2 =='Base_A':
            
            
            
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0001.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0002.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0003.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_02.csv')
                #df = df1
                model = models3
                scaler = scaler3

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_03.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_001.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_002.csv')
                #df = df3
                model = models7
                scaler = scaler7

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_003.csv')
                #df = df3
                model = models7
                scaler = scaler7               
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_02.csv')
                #df = df2
                model = models1
                scaler = scaler1

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
                
                
                
            #st.write(df)
            #st.write(model)
            
            para3['UT6581'] = 0.0
            para3['UT578_A'] = 0.0
            para3['UT578_AF'] = 0.0
            para3['UT578_AS'] = 0.0
            if selected1 == 'UT6581':
                para3['UT6581'] = 1.0
                DT = DT[DT['UT6581']==1]
                DT = DT[DT['Base_A']==1]
            if selected1 == 'UT578_A':
                para3['UT578_A'] = 1.0
                DT = DT[DT['UT578_A']==1]
                DT = DT[DT['Base_A']==1]
            if selected1 == 'UT578_AF':
                para3['UT578_AF'] = 1.0
                DT = DT[DT['UT578_AF']==1]
                DT = DT[DT['Base_A']==1]
            if selected1 == 'UT578_AS':
                para3['UT578_AS'] = 1.0
                DT = DT[DT['UT578_AS']==1]
                DT = DT[DT['Base_A']==1]
                
                
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n
            
            DT = DT.reset_index(drop=True)
            
            
            DT['Delta_E'] = 0.0
            
            
            for i in range(DT.shape[0]):
                DT['Delta_E'][i] = ((Target_v['Target_L'].values - DT['Target_L'][i])**2+(Target_v['Target_a'].values - DT['Target_a'][i])**2+(Target_v['Target_b'].values - DT['Target_b'][i])**2)**0.5 
                    
            DT.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            DT = DT.head(3)
            
            
            #st.write('')
            #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.1. 기존 유사 조색제 배합 </h6>", unsafe_allow_html=True)

            st.write(DT)
            
            st.write('')
            st.write('')
            st.write('')
            
            count +=1
            
             
            
             
        if selected2 =='Base_B':
            
            
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0001.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0002.csv')
                #df = df2
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0003.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_01.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_02.csv')
                #df = df1
                model = models4
                scaler = scaler4

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_03.csv')
                #df = df3
                model = models5
                scaler = scaler5
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_001.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_002.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_003.csv')
                #df = df3
                model = models5
                scaler = scaler5
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_01.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_02.csv')
                #df = df2
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
                
                
                
            #st.write(df)
            #st.write(model)
            
            para3['UT6581'] = 0.0
            para3['UT578_A'] = 0.0
            para3['UT578_AF'] = 0.0
            para3['UT578_AS'] = 0.0
            if selected1 == 'UT6581':
                para3['UT6581'] = 1.0
                DT = DT[DT['UT6581']==1]
                DT = DT[DT['Base_B']==1]
            if selected1 == 'UT578_A':
                para3['UT578_A'] = 1.0
                DT = DT[DT['UT578_A']==1]
                DT = DT[DT['Base_B']==1]
            if selected1 == 'UT578_AF':
                para3['UT578_AF'] = 1.0
                DT = DT[DT['UT578_AF']==1]
                DT = DT[DT['Base_B']==1]
            if selected1 == 'UT578_AS':
                para3['UT578_AS'] = 1.0
                DT = DT[DT['UT578_AS']==1]
                DT = DT[DT['Base_B']==1]
                
                
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n
            
            DT = DT.reset_index(drop=True)
            
            
            DT['Delta_E'] = 0.0
            
            
            for i in range(DT.shape[0]):
                DT['Delta_E'][i] = ((Target_v['Target_L'].values - DT['Target_L'][i])**2+(Target_v['Target_a'].values - DT['Target_a'][i])**2+(Target_v['Target_b'].values - DT['Target_b'][i])**2)**0.5 
                    
            DT.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            DT = DT.head(3)
            
            
            #st.write('')
            #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.1. 기존 유사 조색제 배합 </h6>", unsafe_allow_html=True)

            st.write(DT)
            
            st.write('')
            st.write('')
            st.write('')
            
            count +=1

                            


        if selected2 =='Base_C':


            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a2.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a1.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a2.csv')
                #df = df2
                model = models2
                scaler = scaler2
                
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a2.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a1.csv')
                #df = df3
                model = models3
                scaler = scaler3     
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a2.csv')
                #df = df3
                model = models7
                scaler = scaler7
                    
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a1.csv')
                #df = df1
                model = models5
                scaler = scaler5
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a2.csv')
                #df = df1
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
                
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a2.csv')
                #df = df3
                model = models6
                scaler = scaler6
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a2.csv')
                #df = df3
                model = models2
                scaler = scaler2                
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a1.csv')
                #df = df3
                model = models6
                scaler = scaler6
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a2.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a2.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
                
                
                
            #st.write(df)
            #st.write(model)
            
            para3['UT6581'] = 0.0
            para3['UT578_A'] = 0.0
            para3['UT578_AF'] = 0.0
            para3['UT578_AS'] = 0.0
            if selected1 == 'UT6581':
                para3['UT6581'] = 1.0
                DT = DT[DT['UT6581']==1]
                DT = DT[DT['Base_C']==1]
            if selected1 == 'UT578_A':
                para3['UT578_A'] = 1.0
                DT = DT[DT['UT578_A']==1]
                DT = DT[DT['Base_C']==1]
            if selected1 == 'UT578_AF':
                para3['UT578_AF'] = 1.0
                DT = DT[DT['UT578_AF']==1]
                DT = DT[DT['Base_C']==1]
            if selected1 == 'UT578_AS':
                para3['UT578_AS'] = 1.0
                DT = DT[DT['UT578_AS']==1]
                DT = DT[DT['Base_C']==1]
                
                
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n
            
            DT = DT.reset_index(drop=True)
            
            
            DT['Delta_E'] = 0.0
            
            
            for i in range(DT.shape[0]):
                DT['Delta_E'][i] = ((Target_v['Target_L'].values - DT['Target_L'][i])**2+(Target_v['Target_a'].values - DT['Target_a'][i])**2+(Target_v['Target_b'].values - DT['Target_b'][i])**2)**0.5 
                    
            DT.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            DT = DT.head(3)
            
            
            #st.write('')
            #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.1. 기존 유사 조색제 배합 </h6>", unsafe_allow_html=True)

            st.write(DT)
            
            st.write('')
            st.write('')
            st.write('')
            
            count +=1
            
            #para3.to_csv('para3.csv',index=False)
            #DT.to_csv('DT.csv',index=False)
            
            #model.save('model.h5')
            #pickle.dumps(scaler, open('scaler.pkl','wb'))
            #output_model = pickle.dumps(scaler)
            #b64 = base64.b64encode(output_model).decode()
            #f'<a href="data:file/output_model;base64,{b64}" download="scaler.pkl">Download Trained Model .pkl File</a>'
            
            #pickle.load(open('./scaler.pkl', 'wb'))
   
    
    st.write('')     
    st.write('')    
    st.write('')    

    
    st.write('**2.2. 신규 조색 배합 설계**')    
    #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.2. 신규 조색제 배합 설계 </h6>", unsafe_allow_html=True)
    
    ques = st.radio('조색 설계 방법',('기존배합 수정설계','신규배합 자동설계', '신규배합 수동설계'))
    
        
    
    if ques ==  '신규배합 수동설계' :
        
        
        
        color_list2 = ['SK1','SK2','SB1','SB2','SG1','SY1','SY2','SY3','SO1','SP1','SV1','SR1','SR2','SR3']
        
        if selected2 =='Base_A':

            
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0001.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0002.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0003.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_02.csv')
                #df = df1
                model = models3
                scaler = scaler3

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_03.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_001.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_002.csv')
                #df = df3
                model = models7
                scaler = scaler7

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_003.csv')
                #df = df3
                model = models7
                scaler = scaler7               
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_02.csv')
                #df = df2
                model = models1
                scaler = scaler1

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3    
            


        if selected2 =='Base_B':

            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0001.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0002.csv')
                #df = df2
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0003.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_01.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_02.csv')
                #df = df1
                model = models4
                scaler = scaler4

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_03.csv')
                #df = df3
                model = models5
                scaler = scaler5
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_001.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_002.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_003.csv')
                #df = df3
                model = models5
                scaler = scaler5
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_01.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_02.csv')
                #df = df2
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
                
                            



        
        
        if selected2 =='Base_C':
        
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a2.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a1.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a2.csv')
                #df = df2
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a2.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a1.csv')
                #df = df3
                model = models3
                scaler = scaler3     
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a2.csv')
                #df = df3
                model = models7
                scaler = scaler7
                    
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a1.csv')
                #df = df1
                model = models5
                scaler = scaler5
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a2.csv')
                #df = df1
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a2.csv')
                #df = df3
                model = models6
                scaler = scaler6
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a2.csv')
                #df = df3
                model = models2
                scaler = scaler2                
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a1.csv')
                #df = df3
                model = models6
                scaler = scaler6
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a2.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a2.csv')
                #df = df2
                model = models3
                scaler = scaler3
            
            color_list2 = ['SK1','SK2','SB1','SB2','SG1','SY1','SY2','SY3','SO1','SP1','SV1','SR1','SR2','SR3','SW1']
        
        
        #color_list2 = para3[para3.columns != 0.0].columns
        
        #st.write(color_list2)
        
        color_selected = []
        color_selected = st.multiselect("Data Profile : ", color_list2)

        name = []
        test = [] 
        
        for column in color_selected:
            
            max1 = round(float(para3[column].max()),3)
            if max1==0.0 and selected2 =='Base_A': max1=50.0
            if max1==0.0 and selected2 =='Base_B': max1=100.0
            if max1==0.0 and selected2 =='Base_C': max1=200.0
            
            min1 = round(float(para3[column].min()),3)
               
            step = round((max1-min1)/20.0,3)
        
            value = st.slider(column, min1, max1, (min1,max1), step)
                 
            name.append(column)
            test.append(value)



        
            
    if st.button('Run Prediction',key = 2):
        
    
        if ques == '기존배합 수정설계':
            
            
            if selected2 =='Base_A':
                
                
                
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0002.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0003.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_02.csv')
                    #df = df1
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_03.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_002.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_003.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7               
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_02.csv')
                    #df = df2
                    model = models1
                    scaler = scaler1
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                    
                    
                    
                    
                #st.write(df)
                #st.write(model)
                
                para3['UT6581'] = 0.0
                para3['UT578_A'] = 0.0
                para3['UT578_AF'] = 0.0
                para3['UT578_AS'] = 0.0
                if selected1 == 'UT6581':
                    para3['UT6581'] = 1.0
                    DT = DT[DT['UT6581']==1]
                    DT = DT[DT['Base_A']==1]
                if selected1 == 'UT578_A':
                    para3['UT578_A'] = 1.0
                    DT = DT[DT['UT578_A']==1]
                    DT = DT[DT['Base_A']==1]
                if selected1 == 'UT578_AF':
                    para3['UT578_AF'] = 1.0
                    DT = DT[DT['UT578_AF']==1]
                    DT = DT[DT['Base_A']==1]
                if selected1 == 'UT578_AS':
                    para3['UT578_AS'] = 1.0
                    DT = DT[DT['UT578_AS']==1]
                    DT = DT[DT['Base_A']==1]            
            



            if selected2 =='Base_B':
    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0002.csv')
                    #df = df2
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0003.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_01.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_02.csv')
                    #df = df1
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_03.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_002.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_003.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_01.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_02.csv')
                    #df = df2
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                    
                    
                    para3['UT6581'] = 0.0
                    para3['UT578_A'] = 0.0
                    para3['UT578_AF'] = 0.0
                    para3['UT578_AS'] = 0.0
                    if selected1 == 'UT6581':
                        para3['UT6581'] = 1.0
                        DT = DT[DT['UT6581']==1]
                        DT = DT[DT['Base_B']==1]
                    if selected1 == 'UT578_A':
                        para3['UT578_A'] = 1.0
                        DT = DT[DT['UT578_A']==1]
                        DT = DT[DT['Base_B']==1]
                    if selected1 == 'UT578_AF':
                        para3['UT578_AF'] = 1.0
                        DT = DT[DT['UT578_AF']==1]
                        DT = DT[DT['Base_B']==1]
                    if selected1 == 'UT578_AS':
                        para3['UT578_AS'] = 1.0
                        DT = DT[DT['UT578_AS']==1]
                        DT = DT[DT['Base_B']==1]            
                
                



            

            
            if selected2 =='Base_C':
                
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a2.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a1.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a2.csv')
                    #df = df2
                    model = models2
                    scaler = scaler2
                    
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a2.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3     
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a2.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
                        
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a1.csv')
                    #df = df1
                    model = models5
                    scaler = scaler5
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a2.csv')
                    #df = df1
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a2.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2                
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a1.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a2.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a2.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                                            
                    
                
                para3['UT6581'] = 0.0
                para3['UT578_A'] = 0.0
                para3['UT578_AF'] = 0.0
                para3['UT578_AS'] = 0.0
                if selected1 == 'UT6581':
                    para3['UT6581'] = 1.0
                    DT = DT[DT['UT6581']==1]
                    DT = DT[DT['Base_C']==1]
                if selected1 == 'UT578_A':
                    para3['UT578_A'] = 1.0
                    DT = DT[DT['UT578_A']==1]
                    DT = DT[DT['Base_C']==1]
                if selected1 == 'UT578_AF':
                    para3['UT578_AF'] = 1.0
                    DT = DT[DT['UT578_AF']==1]
                    DT = DT[DT['Base_C']==1]
                if selected1 == 'UT578_AS':
                    para3['UT578_AS'] = 1.0
                    DT = DT[DT['UT578_AS']==1]
                    DT = DT[DT['Base_C']==1]
                
                
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n
            
            
            DT = DT.reset_index(drop=True)
            
            DT['Delta_E'] = 0.0
            
            
            for i in range(DT.shape[0]):
                DT['Delta_E'][i] = ((Target_v['Target_L'].values - DT['Target_L'][i])**2+(Target_v['Target_a'].values - DT['Target_a'][i])**2+(Target_v['Target_b'].values - DT['Target_b'][i])**2)**0.5 
                    
            DT.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            DT = DT.head(3)
            
            DT = DT.reset_index(drop=True)
            
            
            st.write('')
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 기존 유사 조색 배합 </h6>", unsafe_allow_html=True)

            st.write(DT)
            
            
            DT2 = DT.drop(['Target_L','Target_a','Target_b','Delta_E'], axis=1)
            
            datafile = DT2.values
            
            rescaleddatafile = scaler.transform(datafile)

            predictions2 = model.predict(rescaleddatafile)
            
            predictions2 = pd.DataFrame(predictions2, columns=['Pred_L','Pred_a','Pred_b'])
                
            para4 = pd.concat([DT2,predictions2], axis=1)
               
            para4 = para4.reset_index(drop=True)
            
            #st.write(para4)
            
            
            NTarget_v = []
            NTarget_v.append(Target_v['Target_L']+(para4['Pred_L'][0]-DT['Target_L'][0]))
            NTarget_v.append(Target_v['Target_a']+(para4['Pred_a'][0]-DT['Target_a'][0]))
            NTarget_v.append(Target_v['Target_b']+(para4['Pred_b'][0]-DT['Target_b'][0]))
            
            
            NTarget_v = pd.DataFrame(NTarget_v)
            NTarget_v = NTarget_v.T
            NTarget_v.columns = Target_n
            
            Diff = Target_v - NTarget_v
            
            #st.write(Diff)
            
            #st.write(NTarget_v)
            
            df_min0 = DT.iloc[0]
            df_min1 = DT.iloc[1]
            df_min2 = DT.iloc[2]
            
            #df_min0 = pd.DataFrame(df_min0)
            #test = df_min0[df_min0 !=0]
            #st.write(test)
            #st.write(df_min0)
            #st.write(df_min0.shape)
    
            
            #st.write(df_min0)
            #st.write(df_min0.shape())
            
            para2 =[]
            para6 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min0.shape[0]-11):
                    
                column = df_min0.index[j]
                
     
                    
                if df_min0.iloc[j] > 0:
                    min = round(df_min0.iloc[j] - 20,0)
                    if min <0: min = 0 
                    max = round(df_min0.iloc[j] + 20,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/40.0)  
                    col_list.append(column)
                    para2.append(para)
                                          
            para2 = pd.DataFrame(para2)
            para2 = para2.T
            #st.write(col_list)
            para6 = para2
            para6.columns = col_list
            
            #st.write(para6)
    
    
                
            
            para21 =[]
            para61 = pd.DataFrame()
    
            col_list1 = []
            for j in range(df_min1.shape[0]-11):
                    
                column = df_min1.index[j]
    
                    
                    
                if df_min1.iloc[j] > 0:
                    min = round(df_min1.iloc[j] - 20,0)
                    if min <0: min = 0 
                    max = round(df_min1.iloc[j] + 20,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/40.0)  
                    col_list1.append(column)
                    para21.append(para)
                        
            para21 = pd.DataFrame(para21)
            para21 = para21.T
            #st.write(col_list1)
            para61 = para21
            para61.columns = col_list1
            #st.write(para61)
    
            
            
            para22 =[]
            para62 = pd.DataFrame()
            col_list2 = []
            for j in range(df_min2.shape[0]-11):
                    
                column = df_min2.index[j]
                
                    
                if df_min2.iloc[j] > 0:
                    min = round(df_min2.iloc[j] - 20,0)
                    if min <0: min = 0 
                    max = round(df_min2.iloc[j] + 20,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/20.0)  
                    col_list2.append(column)
                    para22.append(para)
                        
            para22 = pd.DataFrame(para22)
            para22 = para22.T
            #st.write(col_list1)
            para62 = para22
            para62.columns = col_list2
            #st.write(para62)
    
            
            
                
                
            
            
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
    
            para7 = []
            for i in range(3000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para6.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para6[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para7.append(para5)
                       
    
            para7 = pd.DataFrame(para7, columns=X.columns) 
            
            
            
            para71 = []
            
            for i in range(2500):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para61.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para61[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para71.append(para5)
                       
    
            para71 = pd.DataFrame(para71, columns=X.columns) 
            
            
            
            para72 = []
            
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para62.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para62[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para72.append(para5)
                       
            para72 = pd.DataFrame(para72, columns=X.columns)
            
            
    
    
            
            #para7 = para7.drop_duplicates()
            #para7 = para7.reset_index(drop=True)
            
    
            datafile2 = para7.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para7 = pd.concat([para7,predictions3], axis=1)
                   
            para7 = para7.reset_index(drop = True)
            
            para7['Delta_E'] = 0.0
                              
            for i in range(para7.shape[0]):
    
                para7['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            para7 = para7.head(1)
            para77 = para4[:1]
            
            
            para7 = pd.concat([para7,para77], axis=0)
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para7 = para7.head(1)
    
            #st.write(para7)
            
            
            
            
            datafile2 = para71.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para71 = pd.concat([para71,predictions3], axis=1)
                   
            para71 = para71.reset_index(drop = True)
            
            para71['Delta_E'] = 0.0
                              
            for i in range(para71.shape[0]):
    
                para71['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para71.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para71 = para71.head(1)
            para717 = para4[1:2]
            
            
            para71 = pd.concat([para71,para717], axis=0)
            para71.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para71 = para71.head(1)
            #st.write(para71)
            
            
            
            
            datafile2 = para72.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para72 = pd.concat([para72,predictions3], axis=1)
                   
            para72 = para72.reset_index(drop = True)
            
            para72['Delta_E'] = 0.0
                              
            for i in range(para72.shape[0]):
    
                para72['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para72.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para72 = para72.head(1)
            para727 = para4[2:3]
            
            para72 = pd.concat([para72,para727], axis=0)
            para72.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para72 = para72.head(1)
            #st.write(para72)
            
            
            
            #st.write('**2차 선정 조색제 배합:**')
            
            para7 = pd.concat([para7,para71,para72], axis=0)
            
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para7 = para7.reset_index(drop=True)
            
            #st.write(para7)
            
            
            """st.write('**2차 선정 조색제 배합:**')
            
            df_min0 = para4.head(3)
            df_min1 = para7.head(3)
            
            df_min11 = pd.concat([df_min0,df_min1], axis=0)
            
            df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            
            st.write(para7)"""
            
            
            
            
            df_min3 = para7.iloc[0]
            df_min4 = para7.iloc[1]
            df_min5 = para7.iloc[2]
            
            
    
            para23 =[]
            para63 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min3.shape[0]-11):
                    
                column = df_min3.index[j]
                
                    
                if df_min3.iloc[j] > 0:
                    min = round(df_min3.iloc[j] - 10,0)
                    if min <0: min = 0 
                    max = round(df_min3.iloc[j] + 10,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/20.0)  
                    col_list.append(column)
                    para23.append(para)
                                          
            para23 = pd.DataFrame(para23)
            para23 = para23.T
            #st.write(col_list)
            para63 = para23
            para63.columns = col_list
            
            
            
            para24 =[]
            para64 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min4.shape[0]-11):
                    
                column = df_min4.index[j]
                
                    
                if df_min4.iloc[j] > 0:
                    min = round(df_min4.iloc[j] - 10,0)
                    if min <0: min = 0 
                    max = round(df_min4.iloc[j] + 10,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/20.0)  
                    col_list.append(column)
                    para24.append(para)
                                          
            para24 = pd.DataFrame(para24)
            para24 = para24.T
            #st.write(col_list)
            para64 = para24
            para64.columns = col_list
            
            
            
            para25 =[]
            para65 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min5.shape[0]-11):
                    
                column = df_min5.index[j]
                
                    
                if df_min5.iloc[j] > 0:
                    min = round(df_min5.iloc[j] - 10,0)
                    if min <0: min = 0 
                    max = round(df_min5.iloc[j] + 10,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/10.0)  
                    col_list.append(column)
                    para25.append(para)
                                          
            para25 = pd.DataFrame(para25)
            para25 = para25.T
            #st.write(col_list)
            para65 = para25
            para65.columns = col_list
            
     
    
                
            #st.write(para61)
            
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
            para8 = []
            for i in range(2000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para63.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para63[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para8.append(para5)
                       
    
            para8 = pd.DataFrame(para8, columns=X.columns) 
            
            
            para81 = []
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para64.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para64[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para81.append(para5)
                       
    
            para81 = pd.DataFrame(para81, columns=X.columns) 
            
            
            para82 = []
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para65.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para65[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para82.append(para5)
                       
    
            para82 = pd.DataFrame(para82, columns=X.columns) 
            
    
    
            datafile3 = para8.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para8 = pd.concat([para8,predictions4], axis=1)
               
            para8 = para8.reset_index(drop = True)
            
            para8['Delta_E'] = 0.0
                              
            for i in range(para8.shape[0]):
    
                para8['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(1)
            para88 = para7[:1]
            
            
            para8 = pd.concat([para8,para88], axis=0)
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(1)
    
            #st.write(para8)
            
            
            
            datafile3 = para81.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para81 = pd.concat([para81,predictions4], axis=1)
               
            para81 = para81.reset_index(drop = True)
            
            para81['Delta_E'] = 0.0
                              
            for i in range(para81.shape[0]):
    
                para81['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para81 = para81.head(1)
            para818 = para7[1:2]
            
            
            para81 = pd.concat([para81,para818], axis=0)
            para81.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para81 = para81.head(1)
    
            #st.write(para81)
            
            
            
            datafile3 = para82.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para82 = pd.concat([para82,predictions4], axis=1)
               
            para82 = para82.reset_index(drop = True)
            
            para82['Delta_E'] = 0.0
                              
            for i in range(para82.shape[0]):
    
                para82['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para82 = para82.head(1)
            para828 = para7[2:3]
            
            
            para82 = pd.concat([para82,para828], axis=0)
            para82.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para82 = para82.head(1)
    
            #st.write(para82)
            
            
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 신규 수정 조색 배합 </h6>", unsafe_allow_html=True)
                        
            para8 = pd.concat([para8,para81,para82], axis=0)
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.reset_index(drop=True)
            
            for i in range(para8.shape[0]):
                para8['Pred_L'][i] = para8['Pred_L'][i] + Diff['Target_L']
                para8['Pred_a'][i] = para8['Pred_a'][i] + Diff['Target_a']
                para8['Pred_b'][i] = para8['Pred_b'][i] + Diff['Target_b']

            
            st.write(para8)
            
            
            
            st.write("")  
            
            """#df_min = para4.head(3)
            df_min1 = para7.head(3)
            df_min2 = para8.head(3)
            
            df_min3 = pd.concat([df_min1,df_min2], axis=0)
            df_min3.sort_values(by='Delta_E', ascending=True, inplace =True)
            df_min3 = df_min3.reset_index(drop=True)
            st.write('**최종 선정 조색제 배합 :**')
            st.write(df_min3)"""
            
            
    
                       #st.info(list(Selected_X2))
                       
            """st.write('')
            st.write('**Total results:**')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.scatterplot(x=range(para7.shape[0]),y=para7.iloc[:,-1],s=30,color='red')
            st.pyplot()"""
               
            st.write('')
            st.write('')    


            
            
            
            
            
            
            
            
            
            
            
            
            
        if ques == '신규배합 자동설계':
            
            

            st.markdown("<h6 style='text-align: left; color: darkblue;'> 2. 배합 샘플 생성  </h6>", unsafe_allow_html=True)
            
            if selected2 =='Base_A':
                
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0002.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0003.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_02.csv')
                    #df = df1
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_03.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_002.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_003.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7               
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_02.csv')
                    #df = df2
                    model = models1
                    scaler = scaler1
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3                
                


            if selected2 =='Base_B':
    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0002.csv')
                    #df = df2
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0003.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_01.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_02.csv')
                    #df = df1
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_03.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_002.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_003.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_01.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_02.csv')
                    #df = df2
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3

                
                
            
            
            if selected2 =='Base_C':
            
            
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a2.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a1.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a2.csv')
                    #df = df2
                    model = models2
                    scaler = scaler2
                    
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a2.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3     
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a2.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
                        
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a1.csv')
                    #df = df1
                    model = models5
                    scaler = scaler5
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a2.csv')
                    #df = df1
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a2.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2                
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a1.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a2.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a2.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                                   
            
                
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 조색 배합 샘플 </h6>", unsafe_allow_html=True)
            
            st.write(para3)
            
            para3 = para3.drop(['sum'], axis=1)

            datafile = para3.values
            
            
            rescaleddatafile = scaler.transform(datafile)
               
               
            predictions2 = model.predict(rescaleddatafile)
       
            predictions2 = pd.DataFrame(predictions2, columns=['Pred_L','Pred_a','Pred_b'])
               
                
            para4 = pd.concat([para3,predictions2], axis=1)

               
            para4 = para4.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n


            para4['Delta_E'] = 0.0
            for i in range(para4.shape[0]):
                para4['Delta_E'][i] = ((Target_v['Target_L'] - predictions2['Pred_L'][i])**2+(Target_v['Target_a'] - predictions2['Pred_a'][i])**2+(Target_v['Target_b'] - predictions2['Pred_b'][i])**2)**0.5 

            para4.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            #check = para4[para4['Delta_E']<1.5].count()
            
            #st.write(check[0])


        
            st.write('')
            st.write('')
    

            
            
            #st.markdown("<h6 style='text-align: left; color: darkblue;'> 3. 조색 배합 예측 결과 </h6>", unsafe_allow_html=True)
            
            #st.write(para4)
                   
            df_min = para4.head(10)
            
            #st.write(df_min)
            
            check = []
            for i in range(df_min.shape[0]):
                
                test = df_min.iloc[i]
                test = test[test !=0]
                              
                check.append(test.index)
                
                
            check = pd.DataFrame(check)  
            check = check.drop_duplicates()
            
            list1 = list(check.index)
            #st.write(list1)
            new_df = []
            for i in list1:
                #st.write(i)
                new_df.append(df_min.iloc[i])
                
            #st.write(check)
            
            new_df = pd.DataFrame(new_df)
            #st.write(new_df)
            
            new_df = new_df.reset_index(drop=True)
            st.write('')
    
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 1차 선정 조색 배합 </h6>", unsafe_allow_html=True)
            #st.write('**1차 선정 조색제 배합:**')
            st.write(new_df)
            
            df_min = new_df.head(3)
            
            
    
            st.write('')
            st.write('')
            
            df_min0 = df_min.iloc[0]
            df_min1 = df_min.iloc[1]
            df_min2 = df_min.iloc[2]
            
            #df_min0 = pd.DataFrame(df_min0)
            #test = df_min0[df_min0 !=0]
            #st.write(test)
            #st.write(df_min0)
            #st.write(df_min0.shape)
    
            
            
            para2 =[]
            para6 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min0.shape[0]-11):
                    
                column = df_min0.index[j]
                
     
                    
                if df_min0.iloc[j] > 0:
                    min = round(df_min0.iloc[j] - 50,0)
                    if min <0: min = 0 
                    max = round(df_min0.iloc[j] + 50,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/20.0)  
                    col_list.append(column)
                    para2.append(para)
                                          
            para2 = pd.DataFrame(para2)
            para2 = para2.T
            #st.write(col_list)
            para6 = para2
            para6.columns = col_list
            
            #st.write(para6)
    
    
                
            
            para21 =[]
            para61 = pd.DataFrame()
    
            col_list1 = []
            for j in range(df_min1.shape[0]-11):
                    
                column = df_min1.index[j]
    
                    
                    
                if df_min1.iloc[j] > 0:
                    min = round(df_min1.iloc[j] - 50,0)
                    if min <0: min = 0 
                    max = round(df_min1.iloc[j] + 50,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/20.0)  
                    col_list1.append(column)
                    para21.append(para)
                        
            para21 = pd.DataFrame(para21)
            para21 = para21.T
            #st.write(col_list1)
            para61 = para21
            para61.columns = col_list1
    
            
            
            para22 =[]
            para62 = pd.DataFrame()
            col_list2 = []
            for j in range(df_min2.shape[0]-11):
                    
                column = df_min2.index[j]
                
                    
                if df_min2.iloc[j] > 0:
                    min = round(df_min2.iloc[j] - 50,0)
                    if min <0: min = 0 
                    max = round(df_min2.iloc[j] + 50,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/10.0)  
                    col_list2.append(column)
                    para22.append(para)
                        
            para22 = pd.DataFrame(para22)
            para22 = para22.T
            #st.write(col_list1)
            para62 = para22
            para62.columns = col_list2
    
            
            
                
                
            #st.write(para61)
            
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
    
            para7 = []
            for i in range(2500):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para6.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para6[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para7.append(para5)
                       
    
            para7 = pd.DataFrame(para7, columns=X.columns) 
            
            
            para71 = []
            
            for i in range(2500):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para61.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para61[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para71.append(para5)
                       
    
            para71 = pd.DataFrame(para71, columns=X.columns) 
            
            
            para72 = []
            
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para62.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para62[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para72.append(para5)
                       
            para72 = pd.DataFrame(para72, columns=X.columns)
            
            
    
    
            
            #para7 = para7.drop_duplicates()
            #para7 = para7.reset_index(drop=True)
            
    
            datafile2 = para7.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para7 = pd.concat([para7,predictions3], axis=1)
                   
            para7 = para7.reset_index(drop = True)
            
            para7['Delta_E'] = 0.0
                              
            for i in range(para7.shape[0]):
    
                para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            para7 = para7.head(1)
            para77 = para4[:1]
            
            
            para7 = pd.concat([para7,para77], axis=0)
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para7 = para7.head(1)
    
            #st.write(para7)
            
            
            
            
            datafile2 = para71.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para71 = pd.concat([para71,predictions3], axis=1)
                   
            para71 = para71.reset_index(drop = True)
            
            para71['Delta_E'] = 0.0
                              
            for i in range(para71.shape[0]):
    
                para71['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para71.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para71 = para71.head(1)
            para717 = para4[1:2]
            
            
            para71 = pd.concat([para71,para717], axis=0)
            para71.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para71 = para71.head(1)
            #st.write(para71)
            
            
            
            
            datafile2 = para72.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para72 = pd.concat([para72,predictions3], axis=1)
                   
            para72 = para72.reset_index(drop = True)
            
            para72['Delta_E'] = 0.0
                              
            for i in range(para72.shape[0]):
    
                para72['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para72.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para72 = para72.head(1)
            para727 = para4[2:3]
            
            para72 = pd.concat([para72,para727], axis=0)
            para72.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para72 = para72.head(1)
            #st.write(para72)
            
            
            
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 2차 선정 조색 배합 </h6>", unsafe_allow_html=True)
            
            para7 = pd.concat([para7,para71,para72], axis=0)
            
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para7 = para7.reset_index(drop=True)
            
            st.write(para7)
            
            
            """st.write('**2차 선정 조색제 배합:**')
            
            df_min0 = para4.head(3)
            df_min1 = para7.head(3)
            
            df_min11 = pd.concat([df_min0,df_min1], axis=0)
            
            df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            
            
            st.write(para7)"""
            
            
            
            
            df_min3 = para7.iloc[0]
            df_min4 = para7.iloc[1]
            df_min5 = para7.iloc[2]
            
            
    
            para23 =[]
            para63 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min3.shape[0]-11):
                    
                column = df_min3.index[j]
                
                    
                if df_min3.iloc[j] > 0:
                    min = round(df_min3.iloc[j] - 5,0)
                    if min <0: min = 0 
                    max = round(df_min3.iloc[j] + 5,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/5.0)  
                    col_list.append(column)
                    para23.append(para)
                                          
            para23 = pd.DataFrame(para23)
            para23 = para23.T
            #st.write(col_list)
            para63 = para23
            para63.columns = col_list
            
            
            
            para24 =[]
            para64 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min4.shape[0]-11):
                    
                column = df_min4.index[j]
                
                    
                if df_min4.iloc[j] > 0:
                    min = round(df_min4.iloc[j] - 10,0)
                    if min <0: min = 0 
                    max = round(df_min4.iloc[j] + 10,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/5.0)  
                    col_list.append(column)
                    para24.append(para)
                                          
            para24 = pd.DataFrame(para24)
            para24 = para24.T
            #st.write(col_list)
            para64 = para24
            para64.columns = col_list
            
            
            
            para25 =[]
            para65 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min5.shape[0]-11):
                    
                column = df_min5.index[j]
                
                    
                if df_min5.iloc[j] > 0:
                    min = round(df_min5.iloc[j] - 20,0)
                    if min <0: min = 0 
                    max = round(df_min5.iloc[j] + 20,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/5.0)  
                    col_list.append(column)
                    para25.append(para)
                                          
            para25 = pd.DataFrame(para25)
            para25 = para25.T
            #st.write(col_list)
            para65 = para25
            para65.columns = col_list
            
     
    
                
            #st.write(para61)
            
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
            para8 = []
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para63.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para63[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para8.append(para5)
                       
    
            para8 = pd.DataFrame(para8, columns=X.columns) 
            
            
            para81 = []
            for i in range(1000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para64.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para64[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para81.append(para5)
                       
    
            para81 = pd.DataFrame(para81, columns=X.columns) 
            
            
            para82 = []
            for i in range(500):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para65.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para65[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para82.append(para5)
                       
    
            para82 = pd.DataFrame(para82, columns=X.columns) 
            
    
    
            datafile3 = para8.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para8 = pd.concat([para8,predictions4], axis=1)
               
            para8 = para8.reset_index(drop = True)
            
            para8['Delta_E'] = 0.0
                              
            for i in range(para8.shape[0]):
    
                para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(1)
            para88 = para7[:1]
            
            
            para8 = pd.concat([para8,para88], axis=0)
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(1)
    
            #st.write(para8)
            
            
            
            datafile3 = para81.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para81 = pd.concat([para81,predictions4], axis=1)
               
            para81 = para81.reset_index(drop = True)
            
            para81['Delta_E'] = 0.0
                              
            for i in range(para81.shape[0]):
    
                para81['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para81 = para81.head(1)
            para818 = para7[1:2]
            
            
            para81 = pd.concat([para81,para818], axis=0)
            para81.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para81 = para81.head(1)
    
            #st.write(para81)
            
            
            
            datafile3 = para82.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para82 = pd.concat([para82,predictions4], axis=1)
               
            para82 = para82.reset_index(drop = True)
            
            para82['Delta_E'] = 0.0
                              
            for i in range(para82.shape[0]):
    
                para82['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para82 = para82.head(1)
            para828 = para7[2:3]
            
            
            para82 = pd.concat([para82,para828], axis=0)
            para82.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para82 = para82.head(1)
    
            #st.write(para82)
            
            st.write('')
            st.write('')
    
            
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 최종 선정 조색 배합 </h6>", unsafe_allow_html=True)
            
            para8 = pd.concat([para8,para81,para82], axis=0)
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.reset_index(drop=True)
            
            st.write(para8)
            
            
            
            st.write("")  
            
            """#df_min = para4.head(3)
            df_min1 = para7.head(3)
            df_min2 = para8.head(3)
            
            df_min3 = pd.concat([df_min1,df_min2], axis=0)
            df_min3.sort_values(by='Delta_E', ascending=True, inplace =True)
            df_min3 = df_min3.reset_index(drop=True)
            st.write('**최종 선정 조색제 배합 :**')
            st.write(df_min3)"""
            
            
    
                       #st.info(list(Selected_X2))
                       
            """st.write('')
            st.write('**Total results:**')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.scatterplot(x=range(para7.shape[0]),y=para7.iloc[:,-1],s=30,color='red')
            st.pyplot()"""
               
            st.write('')
            st.write('')    
       







        
        
        if ques == '신규배합 수동설계':     

            
    
            if selected2 =='Base_A':   

                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0002.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0003.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_02.csv')
                    #df = df1
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata00_03.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_001.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_002.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Adata0_003.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7               
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_01.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_02.csv')
                    #df = df2
                    model = models1
                    scaler = scaler1
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Adata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3                


    
    
            if selected2 =='Base_B':
    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0002.csv')
                    #df = df2
                    model = models4
                    scaler = scaler4
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0003.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_01.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_02.csv')
                    #df = df1
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata00_03.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_001.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_002.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('Bdata0_003.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
                    
    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_01.csv')
                    #df = df3
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_02.csv')
                    #df = df2
                    model = models5
                    scaler = scaler5
    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('Bdata0_0_03.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3


                


            
            if selected2 =='Base_C':            
            
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                    
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0001a2.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a1.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0002a2.csv')
                    #df = df2
                    model = models2
                    scaler = scaler2
                    
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                    
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0003a2.csv')
                    #df = df3
                    model = models4
                    scaler = scaler4
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3     
                if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_01a2.csv')
                    #df = df3
                    model = models7
                    scaler = scaler7
                        
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a1.csv')
                    #df = df1
                    model = models5
                    scaler = scaler5
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_02a2.csv')
                    #df = df1
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data00_03a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                    
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a1.csv')
                    #df = df3
                    model = models1
                    scaler = scaler1
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_001a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a1.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_002a2.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a1.csv')
                    #df = df3
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                    para3 = pd.read_csv('data0_003a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2                
                    
                    
                if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a1.csv')
                    #df = df3
                    model = models6
                    scaler = scaler6
                if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_01a2.csv')
                    #df = df3
                    model = models2
                    scaler = scaler2
                    
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_02a2.csv')
                    #df = df2
                    model = models7
                    scaler = scaler7
                    
                if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a1.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                    para3 = pd.read_csv('data0_0_03a2.csv')
                    #df = df2
                    model = models3
                    scaler = scaler3
                                   
    
    
    
    
            #st.write(name)
            #st.write(test)

            para = []
            para2 = []
            para4 = []
        
            #st.write(test2)
            import itertools
        
            for para in test:
                if para[0] == para[1]:
                    para = itertools.repeat(para[0],100)
                else:
                    para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
            #st.write(para)
                para2.append(para)
        

            #st.write(para2)
        
            para2 = pd.DataFrame(para2)
            para2 = para2.T
            para2 = para2.dropna().reset_index()
            para2.drop(['index'],axis=1,inplace=True)
        
            Iter2 = para2.shape[1]


            for i in range(100):
                para3 = []
                para5 = []
                for j in range(Iter2):
                    #st.write(i,j,list(para2[j]))
                    para3.append(random.sample(list(para2[j]),1))
                    para5.append(para3[j][0])
                
            #para3 = pd.DataFrame(para3).values
            #para4.append(para3)
                para4.append(para5)
            
            
        #para4 = pd.DataFrame(para4)
            para4 = pd.DataFrame(para4)
        
        
            para4.columns=list(name)
        
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 조색 배합 샘플 </h6>", unsafe_allow_html=True)
            #st.write(para4)
        
        
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
    
            para7 = []
            for i in range(6000):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para4.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para4[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para7.append(para5)
                       
    
            para7 = pd.DataFrame(para7, columns=X.columns) 
            
            st.write(para7)



            datafile2 = para7.values
            
            rescaleddatafile2 = scaler.transform(datafile2)
               
            predictions3 = model.predict(rescaleddatafile2)
       
            predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
               
            para7 = pd.concat([para7,predictions3], axis=1)
                   
            para7 = para7.reset_index(drop = True)
            
            para7['Delta_E'] = 0.0
            
            
            Target_v = pd.DataFrame(Target_v)
            Target_v = Target_v.T
            Target_v.columns = Target_n
            
            para7 = para7.reset_index(drop=True)
            
                              
            for i in range(para7.shape[0]):
    
                para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
            
            para7.sort_values(by='Delta_E', ascending=True, inplace =True)
            para7 = para7.head(3)
            
            #st.markdown("<h6 style='text-align: left; color: darkblue;'> 최종 선정 조색 배합 </h6>", unsafe_allow_html=True)
            
            #st.write(para7)

            df_min3 = para7.iloc[0]

            
            
    
            para23 =[]
            para63 = pd.DataFrame()
    
            col_list = []
            for j in range(df_min3.shape[0]-11):
                    
                column = df_min3.index[j]
                
                    
                if df_min3.iloc[j] > 0:
                    min = round(df_min3.iloc[j] - 15,0)
                    if min <0: min = 0 
                    max = round(df_min3.iloc[j] + 15,0)
                    #st.write(max, min)
                    para = np.arange(min, max, (max-min)/30.0)  
                    col_list.append(column)
                    para23.append(para)
                                          
            para23 = pd.DataFrame(para23)
            para23 = para23.T
            #st.write(col_list)
            para63 = para23
            para63.columns = col_list
            

                
            #st.write(para61)
            
            New_x2 = pd.DataFrame(X.iloc[0,:])
            New_x2 = New_x2.T
            
            para8 = []
            for i in range(2500):
                para5 = []
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                                       
                    for col1 in list(para63.columns):
                        if col1 == col:
                            New_x2[col] = random.sample(list(para63[col1]),1)
                                
                    if col == selected1 or col == selected2:
                        New_x2[col] = 1.0
                                                      
                    para5.append(float(New_x2[col].values))
                  
                para8.append(para5)
                       
    
            para8 = pd.DataFrame(para8, columns=X.columns) 
            

    
            datafile3 = para8.values
    
            rescaleddatafile3 = scaler.transform(datafile3)
               
            predictions4 = model.predict(rescaleddatafile3)
       
            predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
               
            para8 = pd.concat([para8,predictions4], axis=1)
               
            para8 = para8.reset_index(drop = True)
            
            para8['Delta_E'] = 0.0
                              
            for i in range(para8.shape[0]):
    
                para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
            
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(1)
            para88 = para7[:1]
            
            
            para8 = pd.concat([para8,para88], axis=0)
            para8.sort_values(by='Delta_E', ascending=True, inplace =True)
            
            para8 = para8.head(3)
            
            st.markdown("<h6 style='text-align: left; color: darkblue;'> 최종 선정 조색제 배합 </h6>", unsafe_allow_html=True)
    
            st.write(para8)
            
            
            
            
            
        

                    
        
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    
    
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')


    
    st.sidebar.header('3. 수정 조색 배합 예측')
    
    st.sidebar.write("<h5 style='text-align: left; color: black;'> 3.1 수정 조색 배합 설계 </h5>", unsafe_allow_html=True)
    

    st.write('')
    st.write('')
    st.write('')
    st.write('')
            

    st.subheader('**3. 수정 조색 배합 예측**')
                

    
    st.write('**3.1. 수정 조색 배합 설계**')
    
    col1,col2 = st.columns([1,1])
    
    with col1:
        select = ['Select','UT578_A','UT578_AS','UT578_AF','UT6581']
        selected3 = st.selectbox("제품 선정 : ", select, key=1)
        
    with col2:
        select = ['Select','Base_A','Base_B','Base_C']
        selected4 = st.selectbox("배합 베이스 선정 : ", select, key=2)
    

    color_list =[]        

    color_list = ['SK1','SK2','SB1','SB2','SG1','SY1','SY2','SY3','SO1','SP1','SV1','SR1','SR2','SR3','SW1']
 
    color_list = pd.DataFrame(color_list,columns=['color'])
    

    #X = df[Selected_X]
    #y = df[Selected_y]
    
        #check = feature_m(X,y)
        #color_list2 = list(check.index[:5])
                  
    colors = st.multiselect('조색제 선택',color_list)
    
    Target_n = []
    Target_v = []

    for color1 in colors:
        value = st.number_input(color1,0.00, 5000.00, 0.0,format="%.2f")
        Target_n.append(color1)
        Target_v.append(value)

    New_x = pd.DataFrame([Target_v],columns=list(Target_n))
    
    
    columns1 = ['Target_L','Target_a','Target_b']
        

    Target_n1 = []
    Target_v1 = []
        
    col1,col2,col3 = st.columns([1,1,1])
        
    with col1:
        value4 = st.number_input(columns1[0], -1000.00, 1000.00, 0.0,format="%.3f", key=1)
        Target_n1.append(columns1[0])
        Target_v1.append(value4)
    with col2:
        value5 = st.number_input(columns1[1], -1000.00, 1000.00, 0.0,format="%.3f", key=2)
        Target_n1.append(columns1[1])
        Target_v1.append(value5)
    with col3:
        value6 = st.number_input(columns1[2], -1000.00, 1000.00, 0.0,format="%.3f", key=3)
        Target_n1.append(columns1[2])
        Target_v1.append(value6)
        
    Target_x = pd.DataFrame([Target_v1],columns=list(Target_n1))
    
    
    columns2 = ['Delta_L','Delta_a','Delta_b']
    
    
    Target_n2 = []
    Target_v2 = []
    
    col1,col2,col3 = st.columns([1,1,1])
        
    with col1:
        value7 = st.number_input(columns2[0], -1000.00, 1000.00, 0.0,format="%.3f", key=5)
        Target_n2.append(columns2[0])
        Target_v2.append(value7)
    with col2:
        value8 = st.number_input(columns2[1], -1000.00, 1000.00, 0.0,format="%.3f", key=6)
        Target_n2.append(columns2[1])
        Target_v2.append(value8)
    with col3:
        value9 = st.number_input(columns2[2], -1000.00, 1000.00, 0.0,format="%.3f", key=7)
        Target_n2.append(columns2[2])
        Target_v2.append(value9)
        
    Delta_x = pd.DataFrame([Target_v2],columns=list(Target_n2))
    
    Actual_x = pd.DataFrame()
    Actual_x['Target_L'] = Target_x['Target_L']  + Delta_x['Delta_L'] 
    Actual_x['Target_a'] = Target_x['Target_a']  + Delta_x['Delta_a'] 
    Actual_x['Target_b'] = Target_x['Target_b']  + Delta_x['Delta_b'] 
    
    
   
                
    New_x2 = pd.DataFrame(X.iloc[0,:])
            
    New_x2 = New_x2.T

    #st.write(New_x2)



    
    col1,col2 = st.columns([1,1])


    if st.button('Run Prediction'): 
        
        
        
            
        for col in New_x2.columns:
            New_x2[col] = 0.0
            for col2 in New_x.columns:
                if col == col2:
                    New_x2[col] = New_x[col2].values
                    
                if col == selected3 or col == selected4:
                    New_x2[col] = 1.0
                    #st.write(col)


        
        
        
        New_x2.index = ['New_case']        
            
        st.write(New_x2.style.format("{:.5}"))
        
        #scaler = StandardScaler().fit(X_train)
        
        rescaledNew_X2 = scaler.transform(New_x2)
    

        predictions = model.predict(rescaledNew_X2)
                    
        predictions = pd.DataFrame(predictions,columns = ['Pred_L','Pred_a','Pred_b'])
            
                
        #st.write('**2.2 색상 예측 결과**')
                    
        #predictions.index = ['Results'] 
            
                                    
        #st.write(predictions.style.format("{:.5}"))
        
        
        Target_v = []
        Target_v.append(predictions['Pred_L'].values)
        Target_v.append(predictions['Pred_a'].values)
        Target_v.append(predictions['Pred_b'].values)
        
        #st.write(Target_v)
        
        
        if selected4 =='Base_A':   
            
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0001.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0002.csv')
                #df = df2
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0003.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_02.csv')
                #df = df1
                model = models3
                scaler = scaler3

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata00_03.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_001.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_002.csv')
                #df = df3
                model = models7
                scaler = scaler7

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Adata0_003.csv')
                #df = df3
                model = models7
                scaler = scaler7               
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_01.csv')
                #df = df3
                model = models3
                scaler = scaler3

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_02.csv')
                #df = df2
                model = models1
                scaler = scaler1

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Adata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3                
                        
            
            
            
        if selected4 =='Base_B':   
        
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0001.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0002.csv')
                #df = df2
                model = models4
                scaler = scaler4
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0003.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_01.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_02.csv')
                #df = df1
                model = models4
                scaler = scaler4

            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata00_03.csv')
                #df = df3
                model = models5
                scaler = scaler5
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_001.csv')
                #df = df3
                model = models4
                scaler = scaler4

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_002.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('Bdata0_003.csv')
                #df = df3
                model = models5
                scaler = scaler5
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_01.csv')
                #df = df3
                model = models5
                scaler = scaler5

            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_02.csv')
                #df = df2
                model = models5
                scaler = scaler5

            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('Bdata0_0_03.csv')
                #df = df2
                model = models3
                scaler = scaler3
        
        
        
        
        
        
        if selected4 =='Base_C':             
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
                
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0001a2.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a1.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0002a2.csv')
                #df = df2
                model = models2
                scaler = scaler2
                
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
                
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0003a2.csv')
                #df = df3
                model = models4
                scaler = scaler4
                
                
                
            if Target_v[0] < 45.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a1.csv')
                #df = df3
                model = models3
                scaler = scaler3     
            if Target_v[0] < 45.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_01a2.csv')
                #df = df3
                model = models7
                scaler = scaler7
                    
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a1.csv')
                #df = df1
                model = models5
                scaler = scaler5
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_02a2.csv')
                #df = df1
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 65.0 and Target_v[1] > 0.0 and Target_v[1] < 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 65.0 and Target_v[1] > 10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data00_03a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
                
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0  and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a1.csv')
                #df = df3
                model = models1
                scaler = scaler1
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_001a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a1.csv')
                #df = df3
                model = models2
                scaler = scaler2
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_002a2.csv')
                #df = df3
                model = models6
                scaler = scaler6
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a1.csv')
                #df = df3
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] > 0.0:
                para3 = pd.read_csv('data0_003a2.csv')
                #df = df3
                model = models2
                scaler = scaler2                
                
                
            if Target_v[0] < 45.0 and Target_v[1] < 0.0  and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a1.csv')
                #df = df3
                model = models6
                scaler = scaler6
            if Target_v[0] < 45.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_01a2.csv')
                #df = df3
                model = models2
                scaler = scaler2
                
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 45.0 and Target_v[0] < 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_02a2.csv')
                #df = df2
                model = models7
                scaler = scaler7
                
            if Target_v[0] > 65.0 and Target_v[1] < 0.0 and Target_v[1] > -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a1.csv')
                #df = df2
                model = models3
                scaler = scaler3
            if Target_v[0] > 65.0 and Target_v[1] < -10.0 and Target_v[2] < 0.0:
                para3 = pd.read_csv('data0_0_03a2.csv')
                #df = df2
                model = models3
                scaler = scaler3
                                   
            
        
        rescaledNew_X2 = scaler.transform(New_x2)
        
        predictions = model.predict(rescaledNew_X2)
                    
        predictions = pd.DataFrame(predictions,columns = ['Pred_L','Pred_a','Pred_b'])
        
        #predictions.index = ['Results'] 
                        
        #st.write(predictions.style.format("{:.5}"))

                

        NTarget_v = []
        NTarget_v.append(Target_x['Target_L'] + (predictions['Pred_L']-Actual_x['Target_L']))
        NTarget_v.append(Target_x['Target_a'] + (predictions['Pred_a']-Actual_x['Target_a']))
        NTarget_v.append(Target_x['Target_b'] + (predictions['Pred_b']-Actual_x['Target_b']))
        
        
        
        NTarget_v = pd.DataFrame(NTarget_v)
        NTarget_v = NTarget_v.T
        NTarget_v.columns = Target_n1
        
        Diff = Target_x - NTarget_v
        
        #st.write(Target_x)
        #st.write(NTarget_v)
        #st.write(Diff)
        #st.write(Actual_x)
        #st.write(predictions)
        
        #Diff = Target_v1 - NTarget_v
        
        #st.write(Diff)
        
        df_min0 = New_x2.T
        
        #df_min0 = pd.DataFrame(df_min0)
        #test = df_min0[df_min0 !=0]
        #st.write(test)
        #st.write(df_min0)
        #st.write(df_min0.shape)
        
        #st.write(df_min0)
        #st.write(df_min0.shape())

        #df_min0 = pd.DataFrame(df_min0)    
        #st.write(df_min0)
        
        para2 =[]
        para6 = pd.DataFrame()

        col_list = []
        for j in range(df_min0.shape[0]-7):
            
            column = df_min0.index[j]
       
            if df_min0.iloc[j].values > 0:
                min1 = float(round(df_min0.iloc[j]-30,0))
                max1 = float(round(df_min0.iloc[j] + 30,0))
                
                if min1 < 0 : min1 = 0
                
                
                para = np.arange(min1, max1, (max1-min1)/20.0)  
                col_list.append(column)
                para2.append(para)
                                      
        para2 = pd.DataFrame(para2)
        para2 = para2.T
        #st.write(col_list)
        para6 = para2
        para6.columns = col_list   
        
        #st.write(para6)
        
 
        
        
        New_x2 = pd.DataFrame(X.iloc[0,:])
        New_x2 = New_x2.T
        
               
        
        para7 = []
        for i in range(6000):
            para5 = []
            for col in New_x2.columns:
                New_x2[col] = 0.0
                                   
                for col1 in list(para6.columns):
                    if col1 == col:
                        New_x2[col] = random.sample(list(para6[col1]),1)
                            
                if col == selected3 or col == selected4:
                    New_x2[col] = 1.0
                                                  
                para5.append(float(New_x2[col].values))
              
            para7.append(para5)
        
        
        para7 = pd.DataFrame(para7,columns=X.columns)
        
        #st.write(para7)
                
        datafile2 = para7.values
        
        rescaleddatafile2 = scaler.transform(datafile2)
           
        predictions3 = model.predict(rescaleddatafile2)
   
        predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
           
        para71 = pd.concat([para7,predictions3], axis=1)
               
        para71 = para71.reset_index(drop = True)
        
        para71['Delta_E'] = 0.0
                          
        for i in range(para71.shape[0]):

            para71['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
        
        para71.sort_values(by='Delta_E', ascending=True, inplace =True)
        
        para71 = para71.head(3)
        
        #st.write(para71)
        
        
        df_min1 = para71.iloc[0]
        
        
        
        para22 =[]
        para62 = pd.DataFrame()

        col_list = []
        for j in range(df_min1.shape[0]-11):
            
            column = df_min1.index[j]
            
       
            if df_min1.iloc[j] > 0:
                min1 = float(round(df_min1.iloc[j] -5,0))
                max1 = float(round(df_min1.iloc[j] +5,0))
                
                if min1 < 0 : min1 = 0
                
                
                para = np.arange(min1, max1, (max1-min1)/10.0)  
                col_list.append(column)
                para22.append(para)
                                      
  
        para22 = pd.DataFrame(para22)
        para22 = para22.T
        #st.write(col_list)
        para62 = para22
        para62.columns = col_list           
        

        
        para72 = []
        for i in range(1000):
            para5 = []
            for col in New_x2.columns:
                New_x2[col] = 0.0
                                   
                for col1 in list(para62.columns):
                    if col1 == col:
                        New_x2[col] = random.sample(list(para62[col1]),1)
                            
                if col == selected3 or col == selected4:
                    New_x2[col] = 1.0
                                                  
                para5.append(float(New_x2[col].values))
              
            para72.append(para5)
        
        
        para72 = pd.DataFrame(para72,columns=X.columns)
        
        #st.write(para72)
                
        datafile3 = para72.values
        
        rescaleddatafile3 = scaler.transform(datafile3)
           
        predictions4 = model.predict(rescaleddatafile3)
   
        predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
           
        para73 = pd.concat([para72,predictions4], axis=1)
               
        para73 = para73.reset_index(drop = True)
        
        para73['Delta_E'] = 0.0
                          
        for i in range(para73.shape[0]):

            para73['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
        
        para73.sort_values(by='Delta_E', ascending=True, inplace =True)
        
        para73 = para73.head(1)        
        

        st.markdown("<h6 style='text-align: left; color: darkblue;'> 최종 선정 조색제 배합 </h6>", unsafe_allow_html=True)


        para73['Pred_L'] = para73['Pred_L'].values + Diff['Target_L'].values
        para73['Pred_a'] = para73['Pred_a'].values + Diff['Target_a'].values
        para73['Pred_b'] = para73['Pred_b'].values + Diff['Target_b'].values
            
        
        
        st.write(para73)
        


 

