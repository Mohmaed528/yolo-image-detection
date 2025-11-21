from ultralytics import YOLO
from PIL import Image,ImageDraw
import streamlit as st


st.set_page_config(layout="wide",page_title='Object Detection')
st.title("Object Detection")
st.info("Object Detection (using YOLO)")

model_choice = st.sidebar.selectbox("Select YOLO Model:",["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
st.sidebar.slider("Detection Confidence:", 0.1, 1.0)

with st.expander(" Recommended Settings Guide"):
    st.markdown("""
    ###  Model Selection
    - **YOLOv8n**: Fastest model, best for low-end devices. Good accuracy.
    - **YOLOv8s**: Balanced speed and accuracy. Recommended for most users.
    - **YOLOv8m**: Higher accuracy, slower. Use for detailed images.

    ###  Confidence Threshold
    - Controls how sure the model must be before detecting an object.
    - **Recommended:** `0.35 – 0.55`
    - Lower values → more detections but may include mistakes.
    - Higher values → fewer detections but more accurate.

    ### ️ Image Tips
    - Use **clear, high-quality images** (minimum 720px).
    - Avoid blurry or low-light images.
    - Prefer **RGB photos** instead of screenshots.
    - Make sure the object is not too small in the image.

    ### ️ Performance Tips
    - Small models (n, s) = faster and smoother.
    - Large models = better accuracy but slower.
    - If detection is slow, use YOLOv8n and reduce image size.

    ### 
    ️ Before You Start
    - Check the image orientation.
    - Set confidence to a comfortable value.
    - Select the proper model based on your device.
    """)


@st.cache_resource

def load_model(model_name):
    return YOLO(model_name)


try:
    model = load_model(model_choice)

except Exception as e:
    st.error(f'Eroor : {e}')



uploaded_file = st.file_uploader("Upload an image(JPG,PNG)",type=['jpg','png','jpeg'])


if uploaded_file is not None:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption='Original Image', use_container_width=True)


    if st.button('Start Detection'):
        st.toast('Processing')
        result = model.predict(image)


        copy_image = image.copy()

        draw = ImageDraw.Draw(copy_image)



        detected_objects = []
        category_count = []

        colors = {}


        for i,box in enumerate(result[0].boxes,1):
            x1,y1,x2,y2 = box.xyxy[0]

            class_name = model.names[int(box.cls)]

            conf = float(box.conf)

            draw.rectangle([x1,y1,x2,y2], outline='red')
            draw.rectangle([x1,y1,x1+30,y1+20], fill='red')
            draw.text((x1 + 3 ,y1 - 3),str(i),fill='white',font_size=22)

            detected_objects.append({
                'ID':i,
                'Object Type':class_name,
                'Confidence':f'{conf:.2f}',

            })
        with col2:
            st.image(copy_image, caption='Final Image', use_container_width=True,channels='RGB')

            detected_objects_count = len(result[0].boxes)


        if detected_objects:
            st.divider()
            st.subheader(f'Detected Objects Details:')
            data = st.dataframe(detected_objects,hide_index=True)




        else:
            st.warning(f'No Object Detected')



st.divider()
st.caption("Developed by Mohamed 🤍")