import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODEL & SCALER
# ============================================================

with open("personality_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# PERSONALITY MAPPING
# ============================================================

personality_mapping = {
    0: "Ambivert",
    1: "Extrovert",
    2: "Introvert"
}


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Personality Type Predictor")

st.write(
    "Discover your personality type using a Machine Learning model."
)

st.info(
    "Rate each question from 1 to 10. "
    "1 = Very Low / Strongly Disagree | "
    "10 = Very High / Strongly Agree"
)

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "👥 Social & Communication",
        "🧠 Thinking & Decisions",
        "🌍 Lifestyle & Interests"
    ]
)


# ============================================================
# TAB 1 - SOCIAL
# ============================================================

with tab1:

    st.subheader("👥 Social & Communication")

    st.caption(
        "Tell us about your social behavior and communication style."
    )

    col1, col2 = st.columns(2)

    with col1:

        social_energy = st.slider(
            "⚡ Social Energy",
            1, 10, 5,
            help="How energetic are you around other people?"
        )

        alone_time_preference = st.slider(
            "🧘 Preference for Alone Time",
            1, 10, 5
        )

        talkativeness = st.slider(
            "💬 Talkativeness",
            1, 10, 5
        )

        group_comfort = st.slider(
            "👥 Comfort in Groups",
            1, 10, 5
        )

        party_liking = st.slider(
            "🎉 Party Liking",
            1, 10, 5
        )

        listening_skill = st.slider(
            "👂 Listening Skill",
            1, 10, 5
        )

    with col2:

        empathy = st.slider(
            "❤️ Empathy",
            1, 10, 5
        )

        friendliness = st.slider(
            "😊 Friendliness",
            1, 10, 5
        )

        public_speaking_comfort = st.slider(
            "🎤 Public Speaking Comfort",
            1, 10, 5
        )

        leadership = st.slider(
            "👑 Leadership",
            1, 10, 5
        )

        work_style_collaborative = st.slider(
            "🤝 Collaborative Work Preference",
            1, 10, 5
        )


# ============================================================
# TAB 2 - THINKING
# ============================================================

with tab2:

    st.subheader("🧠 Thinking & Decision Making")

    st.caption(
        "Tell us about your thinking, planning and decision-making style."
    )

    col1, col2 = st.columns(2)

    with col1:

        deep_reflection = st.slider(
            "🔎 Deep Reflection",
            1, 10, 5
        )

        organization = st.slider(
            "📋 Organization",
            1, 10, 5
        )

        curiosity = st.slider(
            "🔍 Curiosity",
            1, 10, 5
        )

        planning = st.slider(
            "📅 Planning",
            1, 10, 5
        )

    with col2:

        decision_speed = st.slider(
            "⚡ Decision Speed",
            1, 10, 5
        )

        risk_taking = st.slider(
            "🎯 Risk Taking",
            1, 10, 5
        )

        spontaneity = st.slider(
            "✨ Spontaneity",
            1, 10, 5
        )

        routine_preference = st.slider(
            "🔄 Routine Preference",
            1, 10, 5
        )


# ============================================================
# TAB 3 - LIFESTYLE
# ============================================================

with tab3:

    st.subheader("🌍 Lifestyle & Interests")

    st.caption(
        "Tell us about your hobbies, interests and lifestyle."
    )

    col1, col2 = st.columns(2)

    with col1:

        excitement_seeking = st.slider(
            "🔥 Excitement Seeking",
            1, 10, 5
        )

        adventurousness = st.slider(
            "🏔️ Adventurousness",
            1, 10, 5
        )

        reading_habit = st.slider(
            "📚 Reading Habit",
            1, 10, 5
        )

        sports_interest = st.slider(
            "⚽ Sports Interest",
            1, 10, 5
        )

    with col2:

        online_social_usage = st.slider(
            "📱 Online Social Usage",
            1, 10, 5
        )

        travel_desire = st.slider(
            "✈️ Travel Desire",
            1, 10, 5
        )

        gadget_usage = st.slider(
            "💻 Gadget Usage",
            1, 10, 5
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🔮 Predict My Personality",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "social_energy": social_energy,
        "alone_time_preference": alone_time_preference,
        "talkativeness": talkativeness,
        "deep_reflection": deep_reflection,
        "group_comfort": group_comfort,
        "party_liking": party_liking,
        "listening_skill": listening_skill,
        "empathy": empathy,
        "organization": organization,
        "leadership": leadership,
        "risk_taking": risk_taking,
        "public_speaking_comfort": public_speaking_comfort,
        "curiosity": curiosity,
        "routine_preference": routine_preference,
        "excitement_seeking": excitement_seeking,
        "friendliness": friendliness,
        "planning": planning,
        "spontaneity": spontaneity,
        "adventurousness": adventurousness,
        "reading_habit": reading_habit,
        "sports_interest": sports_interest,
        "online_social_usage": online_social_usage,
        "travel_desire": travel_desire,
        "gadget_usage": gadget_usage,
        "work_style_collaborative": work_style_collaborative,
        "decision_speed": decision_speed

    }])


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]

    personality = personality_mapping.get(
        int(prediction),
        str(prediction)
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_scaled)[0]

    predicted_index = list(
        model.classes_
    ).index(prediction)

    confidence = probabilities[predicted_index] * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("🎯 Your Personality Result")

    st.success(
        f"### {personality}"
    )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🧠 Personality Type",
            personality
        )

    with col2:

        st.metric(
            "🎯 Confidence",
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            "🤖 Model",
            "Logistic Regression"
        )


    # ========================================================
    # PROBABILITY SECTION
    # ========================================================

    st.subheader("📊 Personality Probabilities")

    probability_df = pd.DataFrame({

        "Personality": [
            personality_mapping.get(
                int(class_value),
                str(class_value)
            )
            for class_value in model.classes_
        ],

        "Probability": probabilities * 100

    })

    probability_df["Probability"] = (
        probability_df["Probability"].round(2)
    )

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )


    # --------------------------------------------------------
    # PROGRESS BARS
    # --------------------------------------------------------

    for _, row in probability_df.iterrows():

        personality_name = row["Personality"]
        probability = row["Probability"]

        st.write(
            f"**{personality_name}** — {probability:.2f}%"
        )

        st.progress(
            int(probability)
        )


    # ========================================================
    # BAR CHART
    # ========================================================

    st.subheader("📈 Prediction Comparison")

    chart_df = probability_df.set_index(
        "Personality"
    )

    st.bar_chart(
        chart_df["Probability"]
    )


    # ========================================================
    # PERSONALITY EXPLANATION
    # ========================================================

    st.subheader("💡 About Your Personality")

    if personality == "Introvert":

        st.info(
            """
            **Introvert**

            You may prefer quieter environments, independent
            activities and smaller social interactions.
            You may also enjoy spending time alone and reflecting
            on ideas.
            """
        )

    elif personality == "Extrovert":

        st.info(
            """
            **Extrovert**

            You may enjoy social interaction, group activities
            and energetic environments. You may feel comfortable
            communicating and interacting with others.
            """
        )

    elif personality == "Ambivert":

        st.info(
            """
            **Ambivert**

            You may show a balance between introverted and
            extroverted tendencies. Your behavior may change
            depending on the situation and environment.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Personality Type Predictor | "
    "Built with Python, Scikit-learn & Streamlit"
)