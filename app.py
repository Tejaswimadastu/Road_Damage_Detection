import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import pandas as pd


# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(

    page_title="Road Damage Detection",

    page_icon="🚧",

    layout="wide"

)


# -------------------------
# LOAD MODEL
# -------------------------

model = tf.keras.models.load_model(
    "road_damage_model.h5"
)

with open("labels.json") as f:

    labels=json.load(f)



# -------------------------
# HEADER
# -------------------------

st.title(
"🚧 AI-Based Road Damage Detection System"
)

st.subheader(
"Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")



# -------------------------
# ABOUT
# -------------------------

st.header(
"About Project"
)

st.write("""

Road monitoring helps reduce accidents and improve infrastructure safety.

CNN models automatically identify:

• Cracks

• Potholes

• Manholes

Applications:

- Smart Cities
- Highway Monitoring
- Infrastructure Inspection
- Municipal Maintenance

""")


st.markdown("---")



# -------------------------
# IMAGE UPLOAD
# -------------------------

uploaded = st.file_uploader(

"Upload road image",

type=["jpg","jpeg","png","webp"]

)



if uploaded:

    image = Image.open(uploaded)

    st.header(
    "Uploaded Image"
    )

    st.image(

        image,

        use_container_width=True

    )



    # -------------------------
    # PREPROCESSING
    # -------------------------

    img=image.convert("RGB")

    img=img.resize((224,224))

    img=np.array(img)


    img=img/255.0


    img=np.expand_dims(

        img,

        axis=0

    )



    # -------------------------
    # PREDICT
    # -------------------------

    prediction=model.predict(
        img
    )

    pred_class=np.argmax(
        prediction
    )

    confidence=np.max(
        prediction
    )*100



    # -------------------------
    # DAMAGE LOGIC
    # -------------------------

    if confidence < 40:

        damage="No Significant Damage"

        severity="Low"


    else:

        damage=labels[
            str(pred_class)
        ]


        if damage=="Potholes":

            severity="High"

        elif damage=="Cracks":

            severity="Medium"

        else:

            severity="Low"



    st.markdown("---")



    # -------------------------
    # RESULTS
    # -------------------------

    st.header(
    "Prediction Results"
    )


    if damage=="Potholes":

        st.error(

            f"Prediction: {damage}"

        )


    elif damage=="Cracks":

        st.warning(

            f"Prediction: {damage}"

        )


    else:

        st.success(

            f"Prediction: {damage}"

        )



    st.write(

        f"Confidence: {confidence:.2f}%"

    )


    st.write(

        f"Severity Level: {severity}"

    )



    # -------------------------
    # CHART
    # -------------------------

    st.header(
    "Confidence Scores"
    )


    probs=pd.DataFrame({

        "Class":[

            labels[str(i)]

            for i in range(len(labels))

        ],

        "Probability":

        prediction[0]

    })


    st.bar_chart(

        probs.set_index(
            "Class"
        )

    )



    # -------------------------
    # RECOMMENDATIONS
    # -------------------------

    st.header(
    "Recommendations"
    )


    if damage=="Potholes":

        st.error("""

Immediate maintenance recommended.

High accident risk detected.

Repair urgently.

""")



    elif damage=="Cracks":

        st.warning("""

Road crack detected.

Schedule repair soon.

Damage may worsen.

""")


    elif damage=="Manholes":

        st.info("""

Inspect nearby road condition.

Routine maintenance advised.

""")


    else:

        st.success("""

Road appears normal.

No urgent maintenance required.

""")