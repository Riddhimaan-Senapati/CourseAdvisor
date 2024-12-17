import ratemyprofessor
import streamlit as st

#professors = ratemyprofessor.get_school_by_name("University of Massachusetts Amherst")
#selected_professor= st.selectbox('Choose a professor to see their ratings', professors)
st.title("UMass Professor Rankings")
st.caption("See how professors compare")
professor_name = st.text_input("Enter professor name")
#st.write("The current movie title is", professor_name)
school=ratemyprofessor.get_school_by_name("University of Massachusetts Amherst")
professor=ratemyprofessor.get_professor_by_school_and_name(school,professor_name)

if professor is not None:
    st.write("%s works in the %s Department of %s." % (professor.name, professor.department, professor.school.name))
    st.write("Rating: %s / 5.0" % professor.rating)
    st.write("Difficulty: %s / 5.0" % professor.difficulty)
    st.write("Total Ratings: %s" % professor.num_ratings)
    if professor.would_take_again is not None:
        st.write(("Would Take Again: %s" % round(professor.would_take_again, 1)) + '%')
    else:
        st.write("Would Take Again: N/A")


else:
    st.write("Error! Professor not found")


"""
professor = ratemyprofessor.get_professor_by_school_and_name(
    ratemyprofessor.get_school_by_name("University of Massachusetts Amherst"), "Minea")
f professor is not None:
    print("%sworks in the %s Department of %s." % (professor.name, professor.department, professor.school.name))
    print("Rating: %s / 5.0" % professor.rating)
    print("Difficulty: %s / 5.0" % professor.difficulty)
    print("Total Ratings: %s" % professor.num_ratings)
    if professor.would_take_again is not None:
        print(("Would Take Again: %s" % round(professor.would_take_again, 1)) + '%')
    else:
        print("Would Take Again: N/A")

"""
