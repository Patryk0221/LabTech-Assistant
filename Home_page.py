import streamlit as st
img = 'https://raw.githubusercontent.com/Patryk0221/LabTech-Assistant/main/Data/Opening.png'

@st.cache_data
def img_loader(img_url):
    st.image(img_url, use_column_width='always')

if __name__ == '__main__':
    st.set_page_config(page_title='Home', 
                    initial_sidebar_state='auto')
    st.header('LabTech Assistant')

    img_loader(img)
form = st.form('my form')