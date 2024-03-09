import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings("ignore")

# Set the app title
st.title('Bullying Prediction App')

st.write(
    'A machine learning model to predict whether someone will suffer from bullying'
)

# Declare a form to receive the variables to input into the model
form = st.form(key='my_form')
sex = form.radio('What is your genre?', ('Male', 'Female'))
freq_lonely = form.radio('Which frequency do you feel lonely?',
                         ('Never', 'Rarely', 'Sometimes',
                          'Most of the time', 'Always')
                         )
freq_help = form.radio('Which frequency other students are kind and helpful?',
                       ('Never', 'Rarely', 'Sometimes',
                        'Most of the time', 'Always')
                       )
freq_parents_understand = form.radio('Which frequency your parents understand your problems?',
                                     ('Never', 'Rarely', 'Sometimes',
                                      'Most of the time', 'Always')
                                     )
lonely_all_the_time = form.radio('Do you feel lonely most of the time or always?',
                                 ('Yes', 'No')
                                 )
miss_class = form.radio('Do you miss classes without permission?',
                        ('Yes', 'No')
                        )
age = form.radio('How old are you?',
                 ('11 years old or younger', '12 years old', '13 years old',
                  '14 years old', '15 years old', '16 years old',
                  '17 years old', '18 years old or older')
                 )
physically_att = form.radio('How many times were you physically attacked?',
                            ('0 times', '1 time', '2 or 3 times',
                             '4 or 5 times', '6 or 7 times', '8 or 9 times',
                             '10 or 11 times', '12 or more times')
                            )
physically_fight = form.radio('How many times did you physically fight?',
                              ('0 times', '1 time', '2 or 3 times',
                               '4 or 5 times', '6 or 7 times', '8 or 9 times',
                               '10 or 11 times', '12 or more times')
                              )
close_friends = form.radio('How many close friends do you have?',
                           ('0', '1', '2', '3 or more')
                           )
days_missed_school = form.radio('How many days have you already missed school with no permission?',
                                ('0 days', '1 or 2 days', '3 to 5 days',
                                 '6 to 9 days', '10 or more days')
                                )


submit = form.form_submit_button(label='Make Prediction')

# functon to make prediction
@st.cache_resource
def make_prediction(input_data):
    
    # load the model and make prediction
    model = joblib.load('bully.pkl')
    
    data = [input_data]

    df = pd.DataFrame(data=data, columns=['Sex', 
                                          'Felt_lonely', 
                                          'Other_students_kind_and_helpful', 
                                          'Parents_understand_problems', 
                                          'Most_of_the_time_or_always_felt_lonely',
                                          'Missed_classes_or_school_without_permission', 
                                          'Custom_Age',
                                          'Physically_attacked',
                                          'Physical_fighting',
                                          'Close_friends',
                                          'Miss_school_no_permission'])
 
    # make prediction
    result = model.predict(df)
    if result[0] == 1:
        result_class = 'Yes'
    else:
        result_class = 'No'
 
    # check probabilities
    probas = model.predict_proba(df)
    probability = '{:.2f}'.format(float(probas[:, result]))
 
    return result_class, probability

if submit:
    result, probability = make_prediction([sex, freq_lonely, freq_help, freq_parents_understand, lonely_all_the_time, miss_class, age, physically_att, physically_fight, close_friends, days_missed_school])

    st.header('Results')
    st.write('Will suffer from school bullying:', result)
    st.write('Prediction probability:', probability)
