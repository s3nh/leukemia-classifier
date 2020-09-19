import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests 
from PIL import Image 
import io
st.title('Leukemia classification')

url = 'http://192.168.0.107:6007'
endpoint = '/predict/'

st.write('test')
image = st.file_uploader('insert image here')

def process(image, server_url: str):
    
    m = MultipartEncoder(
        fields = {'file' : ('filename', image, 'image/jpeg')} 
    )
    
    r = requests.post(server_url, 
                      data = m, 
                      headers = {'Content-Type' : m.content_type}, 
                      timeout = 8000)
    return r

if st.button('Get classification'):
    if image == None:
        st.write('Insert an image')
       
    else:
        classification = process(image, url + endpoint)
        st.image([image], width = 300)
        
        