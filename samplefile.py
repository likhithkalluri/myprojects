import streamlit as st
import os,requests

upload_folder = 'uploads'

#check if the folder exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


st.header("Upload Image")

upload_file = st.file_uploader("Choose an image...", type=["jpg, jpeg, png"])

if upload_file is not None:
    #get the file name and save path
    file_name = upload_file.name
    save_path = os.path.join(upload_folder, file_name)

    #save the file to a local folder
    with open(save_path, "wb") as f:
        f.write(upload_file.getbuffer())

    st.success(f'image uploaded successfully: {save_path}')

    st.image(upload_file, caption="Uploaded Image", use_column_width=True)

           
    API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    headers = {"Authorization": "Bearer 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"}
    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query(save_path)
    st.write(output)