import pandas as pd
from pandas.io.formats.style_render import Subset
import streamlit as st
from relativeImp import relativeImp
import statsmodels.api as sm

def is_sig(val):
    color = 'green' if val <= 0.05 else 'red'
    return f'background-color: {color}; color: white;'

st.title(':blue[Marquant Driver Analysis App]')

uploaded_file = st.file_uploader('Choose a file')

if uploaded_file is not None:
   df=pd.read_excel(uploaded_file)

   with st.expander('Click to See Input Data'):
       st.write(df)

   st.subheader('Please select the dependent variable:')
   dependent = st.selectbox('Select one variable:', df.columns, key = "dependents")

   st.subheader('Please select the independent variables:')
   independents = st.multiselect('Select the independent variables:', df.columns, key = "independents")

   st.subheader('Please select filter variable if required:')
   filter_var = st.selectbox('Select one variable:', df.columns, key = "filter_var")

   if filter_var is not None:
       all_codes = sorted(df[filter_var].unique())
       filter_codes = st.multiselect('Select the values required for the filter:', df.columns, key = "filter_codes")

   main_body = st.form('main_body')
   submit = main_body.form_submit_button("Calculate")

   if submit:

       yName = dependent
       xNames = independents
       full_list = [yName] + xNames

       # Complete cases only
       df.dropna(axis = 0, how = 'any', inplace = True)

       correls = df[full_list].corr()[yName].tolist()[1:]

       model = sm.OLS(df[yName], sm.add_constant(df[xNames]))
       model_fit = model.fit()
       coeffs = model_fit.params.tolist()[1:]
       pvalues = model_fit.pvalues.tolist()[1:]

       df_results = relativeImp(df, outcomeName = yName, driverNames = xNames)
       r_square = df_results['rawRelaImpt'].sum()       

       df_results.insert(1, 'pvalue', pvalues)
       df_results.insert(1, 'coeff', coeffs)
       df_results.insert(1, 'correls', correls)

       st.title('Driver Analysis Results')       
       st.subheader('Results')
       st.write(f"Base size: {df.shape[0]}")
       st.write(f"R Square: {r_square:.2f}")

       st.dataframe(
           df_results.style.applymap(is_sig, subset=['pvalue']),
           column_config={
               "driver": "Variable",
               "correls": st.column_config.NumberColumn("Correlation",
                                                            format="%.3f"),
               "coeff": st.column_config.NumberColumn("Coefficient",
                                                            format="%.3f"),
               "pvalue": st.column_config.NumberColumn("P-Value",
                                                            format="%.3f"),
               "rawRelaImpt": st.column_config.NumberColumn("Raw Relative Importance",
                                                            format="%.3f"),
               "normRelaImpt": st.column_config.NumberColumn("Standardized Relative Importance",
                                                            format="%.3f"),
               },
           hide_index=True,
           )

else:
    st.warning('you need to upload an excel data file to begin')
