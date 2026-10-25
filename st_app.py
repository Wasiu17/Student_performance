import joblib
import streamlit as st
import pandas as pd
def col_transform(dataframe):
    dataframe['Parental_Involvement'] = dataframe['Parental_Involvement'].map({
        'Low': 1,
        'Medium': 2,
        'High': 3
    })
    
    dataframe['Access_to_Resources'] = dataframe['Access_to_Resources'].map({
        'Low': 1,
        'Medium': 2,
        'High': 3
    })
    
    dataframe['Motivation_Level'] = dataframe['Motivation_Level'].map({
        'Low': 1,
        'Medium': 2,
        'High': 3
    })
    
    dataframe['Extracurricular_Activities'] = dataframe['Extracurricular_Activities'].map({
        'Yes': 1,
        'No': 0
    })
    
    dataframe['Internet_Access'] = dataframe['Internet_Access'].map({
        'Yes': 1,
        'No': 0
    })
    
    dataframe['Family_Income'] = dataframe['Family_Income'].map({
        'Low': 1,
        'Medium': 2,
        'High': 3
    })
    
    dataframe['Teacher_Quality'] = dataframe['Teacher_Quality'].map({
        'Low': 1,
        'Medium': 2,
        'High': 3
    })
    
    dataframe['School_Type'] = dataframe['School_Type'].map({
        'Public': 1,
        'Private': 0
    })
    
    dataframe['Peer_Influence'] = dataframe['Peer_Influence'].map({
        'Positive': 1,
        'Neutral':0,
        'Negative':-1
    })
    
    dataframe['Learning_Disabilities'] = dataframe['Learning_Disabilities'].map({
        'Yes': 1,
        'No': 0
    })
    
    dataframe['Parental_Education_Level'] = dataframe['Parental_Education_Level'].map({
        'High School':1,
        'College':2,
        'Postgraduate':3
    })
    
    dataframe['Distance_from_Home'] = dataframe['Distance_from_Home'].map({
        'Near': 1,
        'Moderate': 2,
        'Far': 3
    })
    
    dataframe['Gender'] = dataframe['Gender'].map({
        'Male': 1,
        'Female': 0
    })

    return dataframe
    

model = joblib.load('Best_model.jb')

col1,col2 = st.columns(2)
with col1:
    Hours_Studied = st.number_input('Study Hours',
                                 min_value=0.0,
                                   value=5.0)
with col2:
    Attendance = st.number_input('Attendance(%)',
                             min_value=0.0,
                             max_value=100.0,
                             value=25.0)
    
col3, col4 = st.columns(2)
with col3:
    Parental_Involvement = st.selectbox('Parent Involvement',
                                    ['Low', 'Medium', 'High'])
with col4:
    Access_to_Resources = st.selectbox('Access to Resources',
                                    ['Low', 'Medium', 'High'])

col5, col6 = st.columns(2)
with col5:
    Extracurricular_Activities = st.selectbox('Extracurricular Activities',
                                    ['Yes', 'No'])
with col6:
    Sleep_Hours = st.number_input('Sleep Hours',
                                 min_value=0,
                                 max_value=10,
                                 value=10)

col7, col8 = st.columns(2)
with col7:
    Previous_Scores = st.number_input('Previous Scores',
                             min_value=0,
                             max_value=100,
                             value=50)

with col8:
    Motivation_Level = st.selectbox('Motivation Level',
                                        ['Low', 'Medium', 'High'])

Internet_Access = st.selectbox('Internet Access',
                                    ['Yes', 'No'])

col9, col10 = st.columns(2)
with col9:
    Tutoring_Sessions = st.number_input('Tutoring_Sessions',
                                        min_value=0,
                                        max_value=8,
                                        value=5)

with col10:
    Family_Income = st.selectbox('Family Income',
                                    ['Low', 'Medium', 'High'])

col11, col12 = st.columns(2)
with col11:
    Teacher_Quality = st.selectbox('Teacher Quality',
                                        ['Low', 'Medium', 'High'])

with col12:
    School_Type = st.selectbox('School Type',
                                ['Public', 'Private'])

col13, col14 = st.columns(2)
with col13:
    Peer_Influence = st.selectbox('Peer Influence',
                                            ['Negative', 'Neutral', 'Positive'])
with col14:
    Physical_Activity = st.number_input('Physical Activity',
                                        min_value=0,
                                        max_value=6,
                                        value=5)

col15, col16 = st.columns(2)
with col15:
    Learning_Disabilities = st.selectbox('Learning Disabilities',
                                    ['Yes', 'No'])

with col16:
    Parental_Education_Level = st.selectbox('Parental Education Level',
                                        ['High School', 'College', 'Postgraduate'])

Distance_from_Home = st.selectbox('Distance from Home',
                                            ['Near', 'Moderate', 'Far'])

Gender = st.selectbox('Gender',
                        ['Male', 'Female'])


if st.button('Predict',):
    input_data = pd.DataFrame([{'Hours_Studied':Hours_Studied,
                                'Attendance':Attendance,
                                'Parental_Involvement':Parental_Involvement,
                                'Access_to_Resources':Access_to_Resources,
                                'Extracurricular_Activities':Extracurricular_Activities,
                                'Sleep_Hours':Sleep_Hours,
                                'Previous_Scores':Previous_Scores,
                                'Motivation_Level':Motivation_Level,
                                'Internet_Access':Internet_Access,
                                'Tutoring_Sessions':Tutoring_Sessions,
                                'Family_Income':Family_Income,
                                'Teacher_Quality':Teacher_Quality,
                                'School_Type':School_Type,
                                'Peer_Influence':Peer_Influence,
                                'Physical_Activity':Physical_Activity,
                                'Learning_Disabilities':Learning_Disabilities,
                                'Parental_Education_Level':Parental_Education_Level,
                                'Distance_from_Home':Distance_from_Home,
                                'Gender':Gender
       }])

    prediction = model.predict(input_data)
    if prediction > 50:
        st.success(f'Your next Score will be {prediction}.')
        st.success(f'You can do Better.')
    else:
        st.warning(f'Your next Score will be {prediction}.')
        st.warning(f'You Need to improve, You can do better')

