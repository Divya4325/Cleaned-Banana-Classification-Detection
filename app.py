import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import time
import os

st.set_page_config(
    page_title="Banana Ripeness Analysis",
    page_icon= "🍌",
    layout="wide"
)

CLASS_NAMES =[    
       "overripe", 
       "ripe", 
       "rotten", 
       "unripe", 
    ]
MODELS_PATH = {
    "CNN":"Models/cnn_banana_model.h5",
    "MOBILENETV2":"Models/mobilenetv2_banana_model.h5",
    "RESNET50":"Models/resnet50_banana_model.h5",
    "EFFICIENTNETB0":"Models/efficientnetb0_banana_model.h5"
}
YOLO_PATH ="Models/Yolo_banana_Detection.pt"
with st.sidebar:
    st.title("🍌 BANANA AI")
    selected = st.radio(
        "Navigation",
        [
            
            "🍌Classification",
            "🔍Object Detection & Counting",
            "📊Model Comparison"
        ]
    )
st.write("Navigation:",selected)    
@st.cache_resource
def load_classification_model(MODELS_PATH):
    model = tf.keras.models.load_model(MODELS_PATH)
    return model
@st.cache_resource
def load_yolo_model():
    model = YOLO(YOLO_PATH)
    return model
def preprocess_image(image,model_name):
    if model_name == "CNN":
        image=image.resize((416,416))
    else:
        image=image.resize((224,224))
    image_array = np.array(image)
    image_array = image_array.astype("float32")/255.0
    image_array = np.expand_dims(image_array,axis=0) 
    return image_array
if selected == "🍌Classification":
    st.title("🍌 Banana Ripeness Classification")
    st.write("Upload an image or capture an image using your camera to classify banana ripeness.")
# Model Selection#
    model_name= st.selectbox(
        "Select Classification Model",
        [
            "CNN",
            "MOBILENETV2",
            "RESNET50",
            "EFFICIENTNETB0"
        ]
    )
    st.info(f"Selected Model:{model_name}")
    model_path= MODELS_PATH[model_name]
    # Check Model Exists
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: {model_path}"
        )         
        st.stop()
    # Load Model
    model= load_classification_model(model_path)
    st.success("Model Loaded Sucessfully!")
    # Image Input #
    input_method = st.radio(
        "Choose Image Input",
        [
            "Upload Image",
            "Camera"
        ],
        horizontal= True
    )    
    image = None
    if input_method == "Upload Image":
       uploaded_file = st.file_uploader(
           "Upload Banana Image",
            type = ["jpg","jpeg","png"]
       ) 
       if uploaded_file:
            image = Image.open(
               uploaded_file
            ).convert("RGB")
    elif input_method == "Camera":
         camera_image = st.camera_input(
             "Take a picture of the Banana"
         )  
         if camera_image:
            image = Image.open(
                camera_image
            ).convert("RGB")
# PREDICTION #
    if image is not None:
       col1,col2 = st.columns(2)
       with col1:
           st.subheader("Input Image")
           st.image(
               image,
               use_container_width= True
           )
       with col2:
            st.subheader("Prediction")
            input_tensor = preprocess_image(image,model_name)
            start_time = time.time()
            predictions = model.predict(
                input_tensor,
                verbose = 0
            )
            inference_time = time.time() - start_time
            probabilites = predictions[0]
            predicted_index= np.argmax(
                probabilites
            )   
            predicted_class = CLASS_NAMES[
               predicted_index
            ]
            confidence = probabilites[
                predicted_index
            ]*100 
            st.success(f"Predicted Class:{predicted_class}") 
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )
            st.metric(
                "Inference Time",
                f"{inference_time:.4f} seconds"
            ) 
            st.subheader(
                "Class Probabilities"
            )
            probability_df = pd.DataFrame(
                {
                    "Class": CLASS_NAMES,
                    "Probability(%)":
                    probabilites * 100
                }
            )
            st.dataframe(
                probability_df,
                use_container_width= True
            )
            st.bar_chart(
                probability_df.set_index("Class")
            )   
# OBJECT DETECTION & COUNTING 
elif selected == "🔍Object Detection & Counting" :
    st.title("🍌 Banana Object Detection & Counting")         
    st.write("Detect Bananas using YOLO and Count the detected objects.")
# Image Input #
    input_method = st.radio(
       "Choose Image Input",
      [
        "Upload Image",
        "Camera"
      ],
      horizontal= True
    )    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Uploade Image",
             type = ["jpg","jpeg","png"]
        )   
        if uploaded_file:
            image = Image.open(
                uploaded_file
            ).convert("RGB")
    else:
        camera_image = st.camera_input(
            "Capture Banana Image"
        )  
        if camera_image:
            image = Image.open(
                camera_image
            ).convert("RGB")
    # YOLO DETECTION #
    if image is not None:
        st.subheader("Input Image")
        st.image(
            image,
            use_container_width= True
        )     
        # LOAD YOLO
        model = load_yolo_model()
        image_array = np.array(image)
        # RUN DETECTION #
        start_time = time.time()
        results = model.predict(
        image_array,
        imgsz= 416,
        conf = 0.25,
        verbose = False
        )    
        inference_time = (
        time.time() - start_time
        )
        result = results[0]
# Annotated Image #
        annotated_frame = result.plot()
        annotated_image = Image.fromarray(
           annotated_frame  
        )    
        st.subheader(
           "Detection Result"
        )
        st.image(
           annotated_image,
           use_container_width= True
        )
# Count Bananas #
        boxes = result.boxes
        banana_count = len(boxes)
        col1,col2 = st.columns(2)
        with col1:
          st.metric(
            "🍌 Banana Count",
            banana_count
           )
        with col2:
           st.metric(
            "Inference Time",
            f"{inference_time:.4f} seconds"
            )    
# Detection Details #
        st.subheader(
           "Detection Details"
        )         
        if banana_count == 0:
           st.warning(
            "No Banana Detected"
           )
        else:
           detection_data = []
           for i,box in enumerate(boxes):
               class_id =int(
               box.cls[0]
            )   
               confidence = float(
               box.conf[0]
            )
               class_name = result.names[
               class_id
            ] 
               detection_data.append(
              {
                "Object":i+1,
                "Class":class_name,
                "Confidence(%)":confidence*100
               }
            )
        detection_df = pd.DataFrame(
               detection_data
            )  
        st.dataframe(
            detection_df,
            use_container_width= True
        )  
#Model Comaparsion #
elif selected == "📊Model Comparison":
    st.title(
        "Banana Classification Model Comparison"
    )   
    st.write(
        "Comparison of the trained Classification Models"
    )
    comparison_data = pd.DataFrame(
        {
            "Model":[
                "CNN",
                "MobileNetV2",
                "ResNet50",
                "EfficientNetB0"
            ],
            "Accuracy": [
                0.83,
                0.89,
                0.88,
                0.96,
            ],
            "Precision": [
                0.84,
                0.89,
                0.89,
                0.96
            ],
            "Recall":[
                0.83,
                0.89,
                0.88,
                0.96
            ],
            "F1-Score":[
                0.83,
                0.89,
                0.88,
                0.96
            ]           
        }
    )    
# Display Table #
    st.subheader(
        "Performance Comparison"
    )   
    st.dataframe(
        comparison_data,
        use_container_width= True
    ) 
# Chart #
    st.subheader("Model Performance")
    chart_data = comparison_data.set_index(
        "Model"
    )         
    st.bar_chart(
        chart_data[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ]
        ]
    )
# Best Model #
    best_model_index = comparison_data[
        "Accuracy"
    ].idxmax()
    best_model = comparison_data.loc[
        best_model_index,
        "Model"
    ]
    best_accuracy = float(
        comparison_data.loc[
           best_model_index,
           "Accuracy"
        ]
    )
    st.success(
        f"Best Classification Model:"
        f"{best_model}"
        f"({best_accuracy*100:.2f}% Accuracy)"
    )
         
         

                