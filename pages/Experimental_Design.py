import streamlit as st
import pyDOE2 as doe
import pandas as pd
import plotly.graph_objects as go

if 'type_of_design' not in st.session_state:    
    st.session_state['type_of_design'] = None
def design():     
    if design_choice == 'Factorial Designs':
        text_factorial = """The Factorial Design method is an ideal choice for experiments where you aim to study the effects of two or more factors simultaneously. In this design, each possible combination of factor levels is considered, enabling a comprehensive analysis of both main effects and interaction effects. [Learn More](http://en.wikipedia.org/wiki/Factorial_experiment)"""

        st.markdown(text_factorial)

        type_of_design = st.selectbox(label='Choose factorial Design', 
                                    options=['General Full-Factorial', '2-level Full-Factorial', '2-level Fractional Factorial', 'Plackett-Burman'],
                                    key = 'type_of_design', index = None)
        
            
    else:
        text_rsm = """Response Surface Design is tailored for experiments where the primary goal is to find the optimal settings of the experimental factors. This approach is particularly useful when you want to model and analyze the relationship between several independent variables and one or more response variables. [Learn More](https://en.wikipedia.org/wiki/Response_surface_methodology)"""

        st.markdown(text_rsm)
        type_of_design = st.selectbox(label='Choose RSM Design', 
                                    options=['Box-Behnken','Central-Composite'],
                                    key = 'type_of_design', index = None)

@st.cache_data
def full_factorial(fact,lvl):
    fact_lvl = [lvl]*fact
    table = doe.fullfact(fact_lvl)
    table = pd.DataFrame(table)
    return table

@st.cache_data
def ff2n(fact):
    table = doe.ff2n(fact)
    table = pd.DataFrame(table)
    return table

@st.cache_data
def fracfact(fractions):
    table = doe.fracfact(fractions)
    table = pd.DataFrame(table)
    return table

@st.cache_data
def pbdesign(num_factors):
    table = doe.pbdesign(num_factors)
    table = pd.DataFrame(table)
    return table

@st.cache_data
def bbdesign(num_factors, center):
    table = doe.bbdesign(num_factors, center)
    table = pd.DataFrame(table)
    return table
@st.cache_data
def ccdesign(num_factors, center, alpha, face):
    table = doe.ccdesign(num_factors, (0, center), alpha, face)
    table = pd.DataFrame(table)
    return table

def choice_of_parameters():
    table = None
    if st.session_state['type_of_design'] == 'General Full-Factorial':
        gff = 'This kind of design offers full flexibility as to the number of discrete levels for each factor in the design'
        st.markdown(gff)
        num_factors, num_levels = st.columns(2)
        chosen_num_factors = num_factors.slider(label='Select number of factors', min_value=1, max_value=10, 
                                                        key='chosen_num_factors')
        chosen_num_levels = num_levels.slider(label='Select number of levels', min_value=2, max_value=5, key='chosen_num_levels', value=None)
        if 'chosen_num_factors' and 'chosen_num_levels' in st.session_state:
                    table = full_factorial(chosen_num_factors, chosen_num_levels)
    elif st.session_state['type_of_design'] == '2-level Full-Factorial': 
        fullf = 'A 2-Level Full-Factorial Design is an experimental framework used to explore the effects of two or more factors, where each factor is limited to two levels. This design method is particularly efficient for understanding how different variables interact and affect the outcome of an experiment'
        st.markdown(fullf)
        chosen_num_factors = st.slider(label='Select number of factors', min_value=1, max_value=10, key='chosen_num_factors', value=None)
        table = ff2n(chosen_num_factors)
    elif st.session_state['type_of_design'] == '2-level Fractional Factorial':
        ff2l = 'We can effectively select a subset of the full-factorial design by intentionally confounding certain main effects of factors with their interaction effects. This process involves creating an alias structure, which symbolically represents these interactions. Alias structures are expressed using notations such as "C = AB", "I = ABC", or "AB = CD". These notations illustrate the relationships between different columns, indicating how one variable or interaction is entwined with others in the design.'
        st.markdown(ff2l)
        fractions = st.text_input('The input to 2-level Fractional Factorial',
                                  value = 'a b ab')
        try:
            table = fracfact(fractions)
        except:
            st.error('Upss... its something wrong with your input')
    elif st.session_state['type_of_design']=='Plackett-Burman':
        pb = 'Plackett-Burman designs offer an alternative approach to creating fractional-factorial designs. Characterized by their expansion in multiples of four, the number of experimental conditions or runs (rows) in these designs increases in a sequence such as 4, 8, 12, and so on. A distinctive feature of these designs is their constraint on the maximum number of factors (columns) that can be accommodated before necessitating an increase in the number of runs. This limit is always one fewer than the subsequent multiple of four in the series.'
        st.write(pb)
        num_factors = st.slider('Select number of factors', min_value=1, max_value=7)
        table = pbdesign(num_factors)
    elif st.session_state['type_of_design'] == 'Box-Behnken':
        bb = 'A Box-Behnken design is a statistical method for the design of experiments, frequently used in response surface methodology. It is particularly effective for fitting quadratic models that require fewer runs than factorial designs. This design is a type of three-level factorial design but does not include all combinations of the levels of factors. Instead, it strategically selects points to efficiently estimate the coefficients of a quadratic model. [Box–Behnken design](https://en.wikipedia.org/wiki/Box%E2%80%93Behnken_design)'
        st.markdown(bb)
        st.image(image='https://raw.githubusercontent.com/Patryk0221/LabTech-Assistant/main/Data/bb.png')
        fact, center = st.columns(2)
        num_factors = fact.slider('Select number of factors', min_value=3, max_value=7, value =3)
        num_center = center.slider('Select number of center points', min_value=1, max_value=7, value=3)
        table = bbdesign(num_factors, num_center)
    elif st.session_state['type_of_design'] == 'Central-Composite':
        ccd = 'The Central Composite Design (CCD) is a widely used experimental design technique in response surface methodology, particularly valuable for building and refining models that involve quadratic terms. This method is designed to efficiently explore the relationships between several independent variables and one or more response variables. [Central composite design](https://en.wikipedia.org/wiki/Central_composite_design).'
        st.markdown(ccd)
        st.image(image='https://raw.githubusercontent.com/Patryk0221/LabTech-Assistant/main/Data/Opening.ccd')
        fact, type = st.columns(2)        
        num_factors = fact.slider('Select number of factors', min_value=1, max_value=7, value =3)
        num_center = fact.slider('Select number of center points', min_value=1, max_value=7, value =3)
        alpha_dict = {'Rotable':'r', 'Orthogonal':'o'}
        face_dict = {'Circumscribed':'ccc', 'Inscribed':'cci', 'Faced':'ccf'}
        alpha = type.selectbox('Select alpha', options=['Rotable', 'Orthogonal'])
        if alpha =='rotable':
            face = type.selectbox('Select face', options = ['Circumscribed', 'Inscribed'] )
        else: 
            face = type.selectbox('Select face', options = ['Circumscribed', 'Inscribed', 'Faced'] )
        table = ccdesign(num_factors, num_center, alpha_dict[alpha], face_dict[face])
        
    else:
        pass
    return table
@st.cache_data
def coded_to_natural(input_table, columns, range_value, center_value):
    for i in columns:   
        input_table[f'natural_{i}'] = input_table[i].apply(lambda x: x * range_value + center_value)
    
    return table

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    st.header('Design of Experiments')
    design_choice = st.selectbox(label='Choose design of interest', 
                                 options=['Factorial Designs', 'Response-Surface Designs'], key='design_choice', index=None)
    if st.session_state['design_choice'] is not None:
        design_f = design()
    else:
        pass
    table = choice_of_parameters()
    keys =[]
    if table is not None:
        columns_rename = []
        for column in table.columns:
            st.text_input(f'Introduce name of factor {column+1}', key=f'name_{column}')
            columns_rename.append(st.session_state[f'name_{column}'])
            keys.append(f'name_{column}')
        table.columns = columns_rename
        st.session_state['columns_rename'] = columns_rename
        
    
    if all(key in st.session_state and st.session_state[key] != "" for key in keys) and table is not None :
        instruction_natural = 'The "zero point" corresponds to the coded value set at level 0, while the "change of value" refers to the unit change from this zero point when moving to a value that is one unit higher or lower.'
        instruction_expander = st.expander('Instruction for coded / natural transformation')
        instruction_expander.markdown(instruction_natural)
        col, val = st.columns(2)
        try:
            for column in table.columns:
                    center_value = col.number_input(label=f'Introduce zero point of {column}', key=f'{column}_center',step=0.1)
                    range_value = val.number_input(label=f'Introduce change of value: {column}', key=f'{column}_value', step=0.1)
                    table[f'natural_{column}'] = table[column].apply(lambda x: x * range_value + center_value)
            if table is not None:
                table_expander = st.expander('Table of design')
                table_expander.write(table)
            csv = convert_df(table)
            st.download_button(label="Download data as CSV", data=csv, file_name='design_df.csv', mime='text/csv')
        except:
            st.error('You introduced invalid columns names')