import streamlit as st
import pandas as pd
import numpy as np
import pickle 
if 'hours_studied' not in st.session_state:
    st.session_state.hours_studied = 0
if 'previous_scores' not in st.session_state:
    st.session_state.previous_scores = 0
if 'extracurricular_activities' not in st.session_state:
    st.session_state.extracurricular_activities = 0
if 'sleep_hours' not in st.session_state:
    st.session_state.sleep_hours = 0
if 'sample_question_papers_practiced' not in st.session_state:
    st.session_state.sample_question_papers_practiced = 0

st.title('Прогнозирование успеваемости студентов')
with st.expander("Описание проекта"):
    st.write('''Прогнозирования успеваемости студентов на основе регрессионной модели''')


number_inputs_container = st.container(border=True)


number_inputs_container.number_input("Часы обучения", key='hours_studied')
number_inputs_container.write(st.session_state.hours_studied)

number_inputs_container.number_input('Предыдущие результаты', key='previous_scores')
number_inputs_container.write(st.session_state.previous_scores)

number_inputs_container.number_input('Внеклассные занятия', key='extracurricular_activities')
number_inputs_container.write(st.session_state.extracurricular_activities)

number_inputs_container.number_input('Часы сна', key='sleep_hours')
number_inputs_container.write(st.session_state.sleep_hours)

number_inputs_container.number_input('Образцы экзаменационных работ', key='sample_question_papers_practiced')
number_inputs_container.write(st.session_state.sample_question_papers_practiced)

model_file_path = "models\project_1_student_performance.sav"
model = pickle.load(open(model_file_path, 'rb'))


def predict_close():  
    input_dataframe = pd.DataFrame({
        'hours_studied' : st.session_state.hours_studied,
        'previous_scores' : st.session_state.previous_scores,
        'extracurricular_activities' : st.session_state.extracurricular_activities,
        'sleep_hours' : st.session_state.sleep_hours,
        'sample_question_papers_practiced' : st.session_state.sample_question_papers_practiced,
        
    }, index=[0])

    input_data = np.log1p(input_dataframe)

    prediction = model.predict(input_data)

    return str(round(*np.expm1(prediction), 2))


def reload():
    del st.session_state.hours_studied
    del st.session_state.previous_scores
    del st.session_state.extracurricular_activities
    del st.session_state.sleep_hours
    del st.session_state.sample_question_papers_practiced


st.button("Сбросить", type="primary", on_click=reload)
if st.button('Предсказать'):
    message = st.chat_message("assistant")
    message.write("Примерная успеваемость студента составляет:")
    message.write(predict_close())
else:  
    message = st.chat_message("assistant")   
    message.write("Ожидание данных для прогнозирования...")
    message.write(".........")