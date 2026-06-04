import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.title {
    text-align: center;
    color: #1f77b4;
}
.result-spam {
    color: red;
    font-size: 24px;
    font-weight: bold;
}
.result-ham {
    color: green;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>📧 Spam Detection System</h1>", unsafe_allow_html=True)

st.write("Enter a message below to check whether it is Spam or Not Spam.")

# User input
message = st.text_area(
    "Message",
    placeholder="Type your email or SMS message here..."
)

# Predict button
if st.button("Detect Spam"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform text
        transformed_message = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(transformed_message)[0]

        st.subheader("Result")

        if prediction == 1:
            st.markdown(
                "<p class='result-spam'>🚨 Spam Message</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p class='result-ham'>✅ Not Spam</p>",
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.caption("Built with Streamlit and Machine Learning")