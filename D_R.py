import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

model = load_model('Digit_Recognizer.hdf5')

st.title('My Digit Recognizer')
st.markdown('''
Write Digit \/ Downhere
''')

SIZE = 192

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#ffffff",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(test_x.reshape(1, 28, 28))
    st.write(f'result: {np.argmax(pred[0])}')
    st.bar_chart(pred[0])
