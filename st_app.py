
import joblib
import streamlit as st
import pandas as pd

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #2563eb);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
    }

    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }

    .result-card {
        background: linear-gradient(120deg, #172554, #2563eb);
        padding: 26px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        min-height: 48px;
    }

    div.stButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    h2, h3 {
        color: #172554;
    }
</style>
""", unsafe_allow_html=True)

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
    

# --------------------------------------------------
# 3. LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Best_model.jb")


try:
    model = load_model()
except Exception as e:
    st.error(
        "Unable to load Best_model.jb. "
        "Check that the model file exists and is compatible."
    )
    st.stop()



# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("Academic Performance Intelligence")

    st.divider()

    st.markdown("### About this tool")
    st.write(
        "Estimate a student's next score using study habits, "
        "attendance, learning resources and other academic factors."
    )

    st.markdown("### How it works")
    st.markdown(
        """
        1. Enter student information.
        2. Run the prediction.
        3. Review the estimated score.
        4. Explore suggestions for improvement.
        """
    )

    st.divider()
    st.caption("AI-assisted estimate • Not a guaranteed result")


# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Dashboard</h1>
    <p>
        Understand academic performance and explore the factors
        that may influence a student's next score.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Student profile")
st.write(
    "Complete the sections below. Your entries will be used "
    "to generate a score estimate."
)


# --------------------------------------------------
# 7. STUDENT INPUT FORM
# --------------------------------------------------
with st.form("student_prediction_form"):

    st.markdown("### 📚 Academic performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        Hours_Studied = st.number_input(
            "Study hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Average hours spent studying."
        )

    with col2:
        Attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75
        )

    with col3:
        Previous_Scores = st.number_input(
            "Previous score",
            min_value=0,
            max_value=100,
            value=50
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        Tutoring_Sessions = st.number_input(
            "Tutoring sessions",
            min_value=0,
            max_value=8,
            value=2
        )

    with col2:
        Sleep_Hours = st.slider(
            "Sleep hours per day",
            min_value=0,
            max_value=10,
            value=7
        )

    with col3:
        Physical_Activity = st.slider(
            "Physical activity (hours)",
            min_value=0,
            max_value=6,
            value=2
        )

    st.divider()
    st.markdown("### 🏠 Learning environment")

    col1, col2 = st.columns(2)

    with col1:
        Parental_Involvement = st.select_slider(
            "Parental involvement",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        Access_to_Resources = st.select_slider(
            "Access to learning resources",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        Internet_Access = st.radio(
            "Internet access",
            ["Yes", "No"],
            horizontal=True
        )

        School_Type = st.radio(
            "School type",
            ["Public", "Private"],
            horizontal=True
        )

        Distance_from_Home = st.selectbox(
            "Distance from home",
            ["Near", "Moderate", "Far"]
        )

    with col2:
        Family_Income = st.select_slider(
            "Family income",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        Teacher_Quality = st.select_slider(
            "Teacher quality",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        Parental_Education_Level = st.selectbox(
            "Parental education",
            ["High School", "College", "Postgraduate"]
        )

        Peer_Influence = st.selectbox(
            "Peer influence",
            ["Negative", "Neutral", "Positive"],
            index=1
        )

    st.divider()
    st.markdown("### 🌱 Personal learning factors")

    col1, col2 = st.columns(2)

    with col1:
        Motivation_Level = st.select_slider(
            "Motivation level",
            options=["Low", "Medium", "High"],
            value="Medium"
        )

        Extracurricular_Activities = st.radio(
            "Participates in extracurricular activities",
            ["Yes", "No"],
            horizontal=True
        )

    with col2:
        Learning_Disabilities = st.radio(
            "Learning disability recorded",
            ["Yes", "No"],
            horizontal=True,
            help="Use this field only when appropriate and with "
                 "respect for student privacy."
        )

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨ Predict student score",
        use_container_width=True
    )


# --------------------------------------------------
# 8. PREDICTION
# --------------------------------------------------
if submitted:

    input_data = pd.DataFrame([{
        "Hours_Studied": Hours_Studied,
        "Attendance": Attendance,
        "Parental_Involvement": Parental_Involvement,
        "Access_to_Resources": Access_to_Resources,
        "Extracurricular_Activities": Extracurricular_Activities,
        "Sleep_Hours": Sleep_Hours,
        "Previous_Scores": Previous_Scores,
        "Motivation_Level": Motivation_Level,
        "Internet_Access": Internet_Access,
        "Tutoring_Sessions": Tutoring_Sessions,
        "Family_Income": Family_Income,
        "Teacher_Quality": Teacher_Quality,
        "School_Type": School_Type,
        "Peer_Influence": Peer_Influence,
        "Physical_Activity": Physical_Activity,
        "Learning_Disabilities": Learning_Disabilities,
        "Parental_Education_Level": Parental_Education_Level,
        "Distance_from_Home": Distance_from_Home,
        "Gender": Gender
    }])

    try:
        # This encoding must match the original model's training pipeline.
        model_input = col_transform(input_data)
        raw_prediction = model.predict(input_data)
        score = float(raw_prediction[0])

        if not pd.notna(score):
            raise ValueError("The model returned an invalid score.")

        st.session_state["student_score"] = score
        st.session_state["student_input"] = input_data.copy()

    except Exception as e:
        st.error(
            "Prediction failed. Check that the model expects these "
            "features and uses the same preprocessing as training."
        )
        st.exception(e)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------
if "student_score" in st.session_state:

    score = st.session_state["student_score"]

    st.divider()
    st.markdown("## 📊 Prediction results")

    result_col, insight_col = st.columns([1, 1.4])

    with result_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="result-card">
                    <h2>Estimated next score</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
            f'<h1 style="font-size:54px;color:#2563eb;">'
            f'{score:.1f}</h1>',
            unsafe_allow_html=True
        )
            st.caption("Model-generated estimate")

            if 0 <= score <= 100:
                st.progress(
                    int(round(score)),
                    text=f"{score:.1f} / 100"
            )
            else:
                st.info(
                "This prediction is outside the 0–100 range. "
                "Review the model's target scale."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with insight_col:
        st.markdown("### Performance insight")

        if score >= 80:
            st.success("Strong estimated performance")
            st.write(
                "The model predicts a strong score. Maintain consistent "
                "study habits and continue reviewing difficult topics."
            )

        elif score >= 60:
            st.info("Promising estimated performance")
            st.write(
                "The estimate suggests a reasonable foundation. "
                "Consistent revision and regular practice may help."
            )

        elif score >= 50:
            st.warning("Room for improvement")
            st.write(
                "Focus on attendance, structured revision and areas "
                "where previous assessments were challenging."
            )

        else:
            st.error("Additional academic support may help")
            st.write(
                "Consider a manageable study timetable, extra practice, "
                "teacher feedback and support tailored to the student's "
                "learning needs."
            )

    st.markdown("### 💡 Recommended next steps")

    recommendations = []

    if Attendance < 85:
        recommendations.append(
            "Improve attendance and establish a consistent class routine."
        )

    if Hours_Studied < 2:
        recommendations.append(
            "Build a regular study habit, starting with short daily sessions."
        )

    if Previous_Scores < 50:
        recommendations.append(
            "Review foundational topics and practise questions from "
            "previous assessments."
        )

    if Tutoring_Sessions < 2:
        recommendations.append(
            "Consider additional teacher support or tutoring where available."
        )

    if Sleep_Hours < 7:
        recommendations.append(
            "Review sleep routines and aim for sufficient, regular rest."
        )

    if Access_to_Resources == "Low":
        recommendations.append(
            "Explore access to textbooks, library materials or other "
            "available learning resources."
        )

    if Motivation_Level == "Low":
        recommendations.append(
            "Set small achievable learning goals and track progress weekly."
        )

    if not recommendations:
        recommendations.append(
            "Maintain a balanced routine and use practice tests to identify "
            "specific topics that need attention."
        )

    for item in recommendations:
        st.markdown(f"- {item}")

    with st.expander("View submitted student profile"):
        st.dataframe(
            st.session_state["student_input"].reset_index(drop=True),
            use_container_width=True
        )

    csv = st.session_state["student_input"].to_csv(index=False)

    st.download_button(
        "⬇️ Download student input as CSV",
        data=csv,
        file_name="student_prediction_input.csv",
        mime="text/csv"
    )

    st.caption(
        "This estimate is not a guarantee of academic outcomes. "
        "Use it alongside teacher observations and actual assessment results."
    )

else:
    st.info(
        "👆 Complete the form above and select **Predict student score** "
        "to see your results here."
    )
