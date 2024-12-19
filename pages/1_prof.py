import ratemyprofessor
import streamlit as st
from components import create_donut_chart, create_percentage_donut


st.title("UMass Professor Rankings")
st.caption("See how professors compare")
professor_name = st.text_input("Enter professor name")
school=ratemyprofessor.get_school_by_name("University of Massachusetts Amherst")
professor=ratemyprofessor.get_professor_by_school_and_name(school,professor_name)

 
if professor is not None:
    # Create four columns for the components
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_donut_chart("Rating", professor.rating)
    
    with col2:
        create_donut_chart("Difficulty", professor.difficulty)

    with col3:
        if professor.would_take_again is not None:
            create_percentage_donut("Would take again %",professor.would_take_again)
        else:
            st.write("Would Take Again: N/A")
    
    with col4:
        st.metric(label="Total Ratings", value=f"{professor.num_ratings}")
    
else:
    st.write("Error! Professor not found")


