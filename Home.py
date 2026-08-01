import streamlit as st
from PIL import Image

from utils.interface import (
    load_model,
    predict,
    CLASS_NAMES
)

st.set_page_config(layout="wide")

st.title("Mobility Assistive Device Detector")

st.write("""
Upload an image containing a mobility aid.

The system will detect:

- Crutches
- Wheel Mobility Aid
- Walkers
""")


@st.cache_resource
def get_model():

    return load_model()


model = get_model()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Run Detection"):

        results = predict(
            model,
            image
        )

        # ====================================================
        # GET ANNOTATED IMAGE
        # ====================================================

        annotated_image = results[0].plot()

        # ====================================================
        # DISPLAY IMAGE (BGR FORMAT)
        # ====================================================

        st.image(
            annotated_image,
            channels="BGR",
            caption="Detection Result",
            use_container_width=True
        )

        boxes = results[0].boxes

        if len(boxes) == 0:

            st.warning(
                "No mobility device detected."
            )

        else:

            st.success(
                "Detection completed."
            )

            st.subheader(
                "Detected Objects"
            )

            for box in boxes:

                class_id = int(box.cls.item())

                confidence = float(
                    box.conf.item()
                )

                class_name = CLASS_NAMES[
                    class_id
                ]

                st.write(
                    f"**{class_name}** : {confidence:.2%}"
                )