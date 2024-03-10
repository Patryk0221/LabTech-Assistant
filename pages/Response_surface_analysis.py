import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.api import OLS
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'df_val' not in st.session_state:
    st.session_state['df_val'] = None


def detect_decimal_separator(sample_string):
    decimal_separators = [',', '.']
    counts = {sep: sample_string.count(sep) for sep in decimal_separators}

    return max(counts, key=counts.get)
    
def detect_csv_separator_and_decimal(file):
    sniffer = csv.Sniffer()
    sample_bytes = file.read(1024)
    file.seek(0)  
    sample_string = sample_bytes.decode('utf-8')

    delimiter = sniffer.sniff(sample_string).delimiter
    decimal_sep = detect_decimal_separator(sample_string)

    return delimiter, decimal_sep

def load_data(uploaded_file):
    if st.session_state['data_frame'] is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                delimiter, decimal_sep = detect_csv_separator_and_decimal(uploaded_file)
                df = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal_sep)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state['df'] = df
            return df
        
        except Exception as e:
            st.error(f"Error: {e}")
            return None
def load_val_data(uploaded_file):
    if st.session_state['val_data_frame'] is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                delimiter, decimal_sep = detect_csv_separator_and_decimal(uploaded_file)
                df_val = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal_sep)
            elif uploaded_file.name.endswith('.xlsx'):
                df_val = pd.read_excel(uploaded_file)
            st.session_state['df_val'] = df_val
            return df_val
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    else:
        pass
def X_selection():
    st.session_state['X'] = selected_X

@st.cache_resource
def poly_transformation(df_X):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_X)
    names = poly.get_feature_names_out()
    X_poly = pd.DataFrame(X_poly,columns=names)
    return X_poly

def create_model(X_poly, df_y, chosen_variables):
    try:
            df_y = df_y.astype(float)
            X_poly = X_poly.astype(float)
            X = X_poly[chosen_variables]
            model_ols = OLS(df_y, X)
            results = model_ols.fit()
            st.session_state['df_y'] = df_y
            st.session_state['model_results'] = results
            st.session_state['selected_poly'] = X
            st.session_state['params'] = results.params
    except Exception as e:
        st.error(f"An error occurred while creating the model: {e}")
        
def clear_model():
    st.session_state['model_results'] = None
    st.session_state['df'] = None
    st.session_state['selected_poly'] = None
    st.session_state['df_y'] = None
    st.session_state["model_X"] = None


def data_grid_rsm(data, x, y, columns, stat_value):
    try:
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        poly = PolynomialFeatures()
        x_range = np.linspace(0, np.max(data[x]), 100)
        y_range = np.linspace(0, np.max(data[y]), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        data_grid = pd.DataFrame({
            x : xx.ravel(),
            y : yy.ravel()
        })
        for col in columns:
            data_grid[col] = stat_value
        data_grid_poly = poly.fit_transform(data_grid)
        names_columns = poly.get_feature_names_out(input_features=data_grid.columns)
        data_grid_poly_df = pd.DataFrame(data_grid_poly, columns=names_columns)
        predicted_values = st.session_state['model_results'].predict(data_grid_poly_df)
        predicted_values_array = predicted_values.to_numpy().reshape(xx.shape)                
        st.session_state['data_grid'] = True
        return xx, yy, predicted_values_array
    except Exception as e:
        st.error(f"An error occurred while creating the model: {e}")
        return None, None, None
def plot_3D(predicted_values_array, xx, yy, y_var_name, x_var_name, x_name, y_name, z_name):
    try:
        custom_colorscale = [
            [0.0, "green"], 
            [0.8, "yellow"],
            [1.0, "red"]
        ]

        fig = go.Figure(data=[go.Surface(z=predicted_values_array, x=xx, y=yy, colorscale=custom_colorscale)])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        fig.update_layout(
            title=f'Model OLS: relation between {x_var_name}, {y_var_name}, and predicted response',
            scene=dict(
                xaxis_title=y_name,
                yaxis_title=x_name,
                zaxis_title=z_name,
                # Nieproporcjonalne zakresy dla każdej osi
                xaxis=dict(range=[np.min(xx), np.max(xx)]),  
                yaxis=dict(range=[np.min(yy), np.max(yy)]),  
                zaxis=dict(range=[np.min(predicted_values_array), np.max(predicted_values_array)]),
                # Ustawienie stałego stosunku aspektu
                aspectmode='cube'
            ),
            autosize=False,
            width=800,
            height=800
        )
        return fig
    except Exception as e:
        error_message = str(e)
        st.error("Wystąpił błąd: ", error_message)
        return None 
import plotly.graph_objects as go

def residual_plot(model, X_poly, Y):
    Y_pred = model.predict(X_poly)
    residuals = Y - Y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Y_pred, y=residuals, mode='markers', name='Residuals'))
    fig.add_trace(go.Scatter(x=Y_pred, y=[0]*len(Y_pred), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Residuals Plot for RSM Model', xaxis_title='Predicted Values', yaxis_title='Residuals', showlegend=True)
    return fig

import plotly.graph_objects as go
import numpy as np

def plot_contour(predicted_values_array, xx, yy, x_name, y_name, z_name):
    try:
        custom_colorscale = [
            [0.0, "green"], 
            [0.8, "yellow"],
            [1.0, "red"]
        ]

        # Tworzenie wykresu konturowego 2D
        fig = go.Figure(data=[
            go.Contour(
                z=predicted_values_array, 
                x=xx[0],  # Użyj wartości krawędzi siatki
                y=yy[:, 0],  # Użyj wartości krawędzi siatki
                colorscale=custom_colorscale
            )
        ])
        fig.update_layout(
            title=f'Relacja pomiędzy {x_name}, {y_name}, a {z_name}',
            xaxis_title=x_name,
            yaxis_title=y_name
        )
        return fig
    except Exception as e:
        st.error(f"Wystąpił błąd: {error_message}")
        return None


def create_dynamic_model(params, variable_bounds, variable_names):
    model = ConcreteModel()

    model.vars = Var(variable_names, bounds=lambda model, i: variable_bounds[i])

    
    def objective_rule(model):
        obj_expr = params.get('1', 0)  
        for var_name in variable_names:
            obj_expr += params.get(var_name, 0) * model.vars[var_name] 
            obj_expr += params.get(f'{var_name}^2', 0) * model.vars[var_name]**2 
        for i, var1 in enumerate(variable_names):
            for var2 in variable_names[i+1:]:
                obj_expr += params.get(f'{var1} {var2}', 0) * model.vars[var1] * model.vars[var2]
        return obj_expr

    model.obj = Objective(rule=objective_rule, sense=maximize)

    return model

def objective_function(x, params, variable_names):
    obj_expr = params.get('1', 0)
    for i, var_name in enumerate(variable_names):
        obj_expr += params.get(var_name, 0) * x[i]
        obj_expr += params.get(f'{var_name}^2', 0) * x[i]**2
    for i, var1 in enumerate(variable_names):
        for j, var2 in enumerate(variable_names[i+1:], i+1):
            obj_expr += params.get(f'{var1} {var2}', 0) * x[i] * x[j]
    return -obj_expr  # Minimize (-f(x)) to maximize f(x)


def prepare_bounds(variable_bounds, variable_names):
    bounds = [variable_bounds[var_name] for var_name in variable_names]
    return bounds


def optimize_with_scipy(params, bounds_dict, variable_names):
    bounds = prepare_bounds(bounds_dict, variable_names)
    initial_guess = np.zeros(len(variable_names))
    optimization = minimize(objective_function, initial_guess, args=(params, variable_names), bounds=bounds, method='SLSQP')
    return optimization
        
def coded_to_natural(input_table, columns, range_value, center_value):
    for i in columns:   
        input_table[f'natural_{i}'] = input_table[i].apply(lambda x: x * range_value + center_value)

tabs = ["Create model", "Model visualisation", "Optimization", 'Validation']
selected_tab = st.sidebar.radio("Chose side", tabs)

def parameters_calculation(val_df):
    try:
        model = st.session_state['model_results']
        X_test_col = st.session_state['model_X']
        y_test_col = st.session_state['model_y']
        X_test = poly_transformation(val_df[X_test_col])
        y_test = val_df[y_test_col]        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred) 
        mea = mean_absolute_error(y_test, y_pred)
        return mse, mea
    except:
        return None, None

if selected_tab == "Create model":
    st.markdown("# Create model")
    with st.container(border=True):
        st.write('The principle of creating a quadratic Response Surface Methodology (RSM) model involves a systematic approach to model and analyze problems where the goal is to optimize an output based on several input variables. This method employs statistical and mathematical techniques to develop an understanding of the relationship between input variables and the response. For a quadratic model, this relationship is characterized using second-degree equations, allowing for the capture of not only the direct effects of input variables on the response but also their interactions and nonlinear effects.')
    st.latex(r'''
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_1^2 + \beta_5x_2^2 + \beta_6x_3^2 
+ \beta_7x_1x_2 + \beta_8x_1x_3 + \beta_9x_2x_3 + \epsilon
''')
    st.markdown("""
Gdzie:
- $y$ jest zmienną zależną (przewidywaną),
- $x_1$, $x_2$, $x_3$ są zmiennymi niezależnymi,
- $\eta_0, \eta_1, ..., \eta_9$ są współczynnikami modelu,
- $\epsilon$ jest błędem losowym modelu.
""")


    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key='data_frame')
    load_data(uploaded_file)
    if st.session_state['data_frame'] is not None:
        df = st.session_state['df']
        expander_df = st.expander('Your data')
        expander_df.write(df)
        with st.form("my_form"):
            st.write('select the columns containing the variables that will be included in the model and the column containing the response value')
            selected_X = st.multiselect('Variables for analysis', options=df.columns, key = 'X')
            selected_y = st.selectbox('Column of response (y)', options=df.columns, index=None, key='y')
            submit_button = st.form_submit_button("Submit")
            
        if submit_button:
            st.session_state['model_X'] = st.session_state['X']
            st.session_state['model_y'] = st.session_state['y']
            
        if st.session_state['y'] is not None and st.session_state['model_results'] is None:
            df_X =df[st.session_state['X']]
            df_y = df[st.session_state['y']] 
            X_poly = poly_transformation(df_X)
            st.write('Performed polynomial transformation of data frame, select variables for model')
            chosen_variables = st.multiselect('Select variables for model', options = X_poly.columns, key ='X_poly')
            create_model_button = st.button("Create model", on_click=create_model, args=(X_poly, df_y, chosen_variables))

    if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
        model_expander = st.expander('Model parameters')
        model_expander.write(st.session_state['model_results'].summary())            
        clear_button = st.button('Clear your model', on_click=clear_model)


if selected_tab == "Model visualisation":
    if 'model_results' not in st.session_state or st.session_state['model_results'] is None:
        st.info('Before visualisation, prepare model')
    else:
        st.markdown('# Model visualisation')
        plot_type = st.selectbox('Chose type of plot', options=['3D Respone surface plot','Contour plot' ,'Residual plot']) 
        df = st.session_state['df']
        if plot_type == '3D Respone surface plot':
            ax_var, cons = st.columns(2)

            y_var_name = ax_var.selectbox('Select y-axis variable', options=df.columns, key='y_var')
            x_var_name = ax_var.selectbox('Select x-axis variable', options=df.columns, key='x_var')
            
            cons_var = [x for x in st.session_state['model_X'] if x not in [y_var_name, x_var_name]]
            cons_val = cons.number_input('Introduce value of constant')
            
            ax_x, ax_y, ax_z = st.columns(3)
            x_name = ax_x.text_input('Introduce x axis name', value=x_var_name)
            y_name= ax_y.text_input('Introduce y axis name', value=y_var_name)
            z_name = ax_z.text_input('Introduce response axis name', value='Response')
            plot_button = st.button('Create plot', key='submit_plot')
            if plot_button:
                xx, yy, predicted_values_array = data_grid_rsm(df, x_var_name, y_var_name, cons_var, cons_val)
                plot = plot_3D(predicted_values_array, xx, yy,  y_var_name, x_var_name, x_name, y_name, z_name)
                st.plotly_chart(plot)
                
        elif plot_type == 'Residual plot':
            resid_plot = residual_plot(model = st.session_state['model_results'], X_poly= st.session_state['selected_poly'],Y=st.session_state['df_y']  )
            st.plotly_chart(resid_plot)
        elif plot_type == 'Contour plot':
            ax_var, cons = st.columns(2)

            y_var_name = ax_var.selectbox('Select y-axis variable', options=df.columns, key='y_var_contour')
            x_var_name = ax_var.selectbox('Select x-axis variable', options=df.columns, key='x_var_contour')
            
            cons_var = [x for x in st.session_state['model_X'] if x not in [y_var_name, x_var_name]]
            cons_val = cons.number_input('Introduce value of constant')
            
            ax_x, ax_y, ax_z = st.columns(3)
            x_name = ax_x.text_input('Introduce x axis name', value=x_var_name)
            y_name= ax_y.text_input('Introduce y axis name', value=y_var_name)
            z_name = ax_z.text_input('Introduce response axis name', value='Response')
            plot_button = st.button('Create plot', key='submit_plot')
            if plot_button:
                xx, yy, predicted_values_array = data_grid_rsm(df, x_var_name, y_var_name, cons_var, cons_val)
                plot_contour = plot_contour(predicted_values_array, xx, yy, x_name, y_name, z_name)
                st.plotly_chart(plot_contour)            
            
        pass
            
            
if selected_tab == "Optimization":
    if 'model_results' not in st.session_state or st.session_state['model_results'] is None:
        st.info('Create model, before optimization')
    else:
        st.markdown('# Optimization')
        st.markdown('### Choice type of your model')
        model_type = st.selectbox('model type', options=['Coded variables', 'Natural variables'])
        info_container = st.container()
        info_container.info('If you used coded variables to create the model, for example (-1, 0, 1) select "Coded variables" while if you used natural variables select "Natural variables"')
        bounds_dict ={}
        
        if model_type == 'Coded variables':
            for i in st.session_state['model_X']:
                lower, upper, zero, change = st.columns(4)
                lower.number_input(f'Lower bound {i}', key = f'lower_bound {i}', value = None)
                upper.number_input(f'Upper bound {i}', key = f'upper_bound {i}', value = None)
                center_value = zero.number_input(label=f'Zero point of {i}', key=f'{i}_center',step=0.1)
                range_value = change.number_input(label=f'Change of : {i}', key=f'{i}_value', step=0.1)
                bounds_dict[i] = (st.session_state[f'lower_bound {i}'], st.session_state[f'upper_bound {i}'])
            optimize_coded_button = st.button('Optimize')
            if optimize_coded_button:
                
                optimization_natural  = optimize_with_scipy(st.session_state['params'], bounds_dict, st.session_state['model_X'])
                optimal_values = optimization_natural.x
                max_obj_value = -optimization_natural.fun
                coded, natural = st.columns(2)
                coded.markdown('#### Optimal parameters (coded)')
                natural.markdown('#### Optimal parameters (natural)')
                
                for i, var_name in enumerate(st.session_state['model_X']):
                    coded.write(f'{var_name}: {optimal_values[i]}')
                    natural.write(f'{var_name}: {optimal_values[i] * range_value + center_value}')
                coded.write(f"Maximum value: {max_obj_value}")
                natural.write(f"Maximum value: {max_obj_value}")
                    
        if model_type == 'Natural variables':
            bounds_dict = {}
            lower_natural, upper_natural = st.columns(2)
            for i in st.session_state['model_X']:
                lower_bound = lower_natural.number_input(f'Lower bound {i}', key = f'lower_bound_natural {i}', value = None)
                upper_bound = upper_natural.number_input(f'Upper bound {i}', key = f'upper_bound_natural {i}', value = None)
                bounds_dict[i] = (st.session_state[f'lower_bound_natural {i}'], st.session_state[f'upper_bound_natural {i}'])
            optimize_natural_button = st.button('Optimize')
            if optimize_natural_button:
                    optimization_natural  = optimize_with_scipy(st.session_state['params'], bounds_dict, st.session_state['model_X'])
                    optimal_values = optimization_natural.x
                    max_obj_value = -optimization_natural.fun
                    for i, var_name in enumerate(st.session_state['model_X']):
                        st.write(f'{var_name}: {optimal_values[i]}')
                    st.write(f"Maximum value: {max_obj_value}")
if selected_tab == "Validation":
    st.markdown('# Model validation')
    if 'model_results' not in st.session_state or st.session_state['model_results'] is None:
        st.info('Create model, before validation')
    else: 
        uploaded_val_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key='val_data_frame')
        load_val_data(uploaded_val_file)
        
    if 'df_val' in st.session_state and st.session_state['df_val'] is not None:
        try:
            val_data_expander = st.expander('Validation data frame')
            val_data_expander.write(st.session_state['df_val'])
            mse, mea = parameters_calculation(st.session_state['df_val'])
            container_results = st.container(border=True)
            container_results.markdown(f"#### Mean Square Error (MSE): {np.round(mse, 2)} ")
            container_results.markdown(f"#### Root Mean Square Error (RMSE): {np.round(np.sqrt(mse), 2)} ")
            container_results.markdown(f"#### Mean Absolute Error (MAE): {np.round(mea, 2)} ")
        except:
            st.error(' Upss... something is wrong, make sure your data contains variables with the names used in the model or the data format is correct')