import streamlit as st
from utils import *
from model import *

from annotated_text import annotated_text
from annotated_text import annotated_text, annotation

st.image("images/article1.jpg", caption="", width=680)
st.title("Topic Prediction")
st.subheader("Created by Mohamed Houssem Beltifa")
st.markdown("--------------------------------------------------")
st.write("text classification model that can automatically categorize news articles into different topics")
st.write("The topics are: Sports, Entertainment, Politics, Technology, Business")

st.markdown("--------------------------------------------------")
st.markdown("--------------------------------------------------")
article=str(st.text_area("Enter your article here : "))
model=themodel()
if article !="":
    predicted_topic = predict_topic(model, article)
    st.markdown("--------------------------------------------------")
    st.markdown("--------------------------------------------------")
    
    # st.write (f"Predicted Topic: {predicted_topic}")
    
    match predicted_topic:
        case "sport":
            annotated_text(
            "Predicted Topic: ",
            annotation("Sports", "", color="#afa", border="1px dashed red"),)
            st.image("images/sport1.jpg", caption="", width=700)

        case "entertainment":
            annotated_text(
            "Predicted Topic: ",
            annotation("Entertainment", "", color="#faa", border="1px dashed red"),)
            st.image("images/entertainment1.jpg", caption="", width=700)

        case "politics":
            annotated_text(
            "Predicted Topic: ",
            annotation("Politics", "", color="#fea", border="1px dashed red"),)
            st.image("images/Politics1.jpg", caption="", width=700)

        case "tech":
            annotated_text(
            "Predicted Topic: ",
            annotation("Technology", "", color="#8ef", border="1px dashed red"),)
            st.image("images/tech1.jpg", caption="", width=700)

        case "business":
            annotated_text(
            "Predicted Topic: ",
            annotation("Business", "", color="#9F8170", border="1px dashed red"),)
            st.image("images/business1.jpg", caption="", width=700)