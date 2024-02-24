import streamlit as st
img = 'Data/opening.png'
@st.cache_data
def img_loader(img):
     st.image(img, use_column_width='always')

if __name__ == '__main__':
    st.set_page_config(page_title='Home', 
                    initial_sidebar_state='auto')
    st.header('LabTech Assistant')

    img_loader(img)
form = st.form('my form')