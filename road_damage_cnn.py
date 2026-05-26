import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# Page settings
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)


# Load CNN model
model = tf.keras.models.load_model(
    "road_damage_model.h5"
)


# Load labels
with open("labels.json","r") as f:
    labels = json.load(f)


# Title
st.title("🚧 Road Damage Detection System")

st.write(
"""
Upload a road image to detect:

• Cracks  
• Potholes  
• Manholes
"""
)


# Upload image
uploaded_file = st.file_uploader(

    "Choose image",

    type=["jpg","jpeg","png","webp"]

)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True

    )


    # Resize image
    img = image.resize((224,224))

    img = np.array(img)


    # Normalize
    img = img / 255.0


    # CNN input shape
    img = np.expand_dims(
        img,
        axis=0
    )


    # Prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100


    damage = labels[
        str(predicted_class)
    ]


    st.success(
        f"Prediction: {damage}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )


    # Probability scores
    st.subheader(
        "Class Probabilities"
    )

    for i,p in enumerate(prediction[0]):

        st.write(

            f"{labels[str(i)]}: {p*100:.2f}%"

        )