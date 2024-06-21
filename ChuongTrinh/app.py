import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.title("Face Mask Detector")
st.sidebar.title("Tùy chọn")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model-003.keras')
    return model

model = load_model()

labels_dict = {0: 'No Mask', 1: 'With mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict_mask(img, model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

st.sidebar.subheader("Tải ảnh lên")
uploaded_file = st.sidebar.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    result_img = detect_and_predict_mask(img_array, model)
    st.image(result_img, caption='Ảnh đã xử lý.', use_column_width=True)

st.sidebar.subheader("Nhận diện thời gian thực")
use_webcam = st.sidebar.checkbox("Sử dụng Webcam")

if use_webcam:
    st.warning("Nhấn 'q' để thoát khỏi webcam.")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_and_predict_mask(frame, model)
        stframe.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
