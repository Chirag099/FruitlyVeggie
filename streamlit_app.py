import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("training_model (1).h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)



#SideBar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page",["Home","About Project","Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("Fruits & Vegetables Recognition System")
    image_path = "image.webp"
    st.image(image_path)


#About Page
elif(app_mode=='About Project'):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

#Prediction Page
elif(app_mode == 'Prediction'):
    st.header("Model Prediction")
    test_image =st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting it's a {}".format(label[result_index]))

