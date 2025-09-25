#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:11:27 2025

An app to analyze dissolved O2 concentration in time. 
Provides options for filtering noise. 

@author: danfeldheim
"""


# Imports
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
# Import a custom user authentication py file
# from users import User
from google.cloud import storage
import requests
from datetime import datetime, timedelta
import io, zipfile
import time
# Import a custom styles file from styles.py
from styles import CSS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.stats import t
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, RANSACRegressor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import math
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import csv
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfMerger
import gc
import psutil
import heapq
from typing import Optional
from PIL import Image


# Clean up any leftover figures or memory at the start of each run
plt.close("all")
gc.collect()

# Checks memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)


class Flow_Control():
    """This class makes all of the calls to other classes and methods."""
    
    def __init__(self):
        
        # Inject custom CSS from styles.py file
        st.markdown(CSS, unsafe_allow_html=True)

        # Get the path relative to the current file (inside Docker container)
        BASE_DIR = os.path.dirname(__file__)
        
        # Create a path to the sqlite user database
        self.db_file = os.path.join(BASE_DIR, "users.db")
         
        
    def all_calls(self):
        """This is the main logic workflow. All calls to other functions are here."""
        
        #---------------------------------------------------------------------------------------------
        # This prevents an error that occurs occasionally when the user attempts
        # to upload a file and the app hasn't loaded fully
        if "app_ready" not in st.session_state:
            
            # Mark app as not ready in state_variables_dict
            # Initialize session state variables
            state_variables_dict = {
                                    "app_ready": False,
                                    "upload_attempted": False,
                                    "authenticated": False,
                                    "login_failed": False,
                                    "smooth_data": False,
                                    "download_options": 'Combined',
                                    "noise_analysis": False,
                                    "residuals_plot": False,
                                    "acf_plot": False,
                                    "newey": False,
                                    "normal": False,
                                    "rolling_regression": False
                                    }
            
            Utilities.add_to_state(state_variables_dict)
    
            # Show loading message for first-time initialization
            st.write('')
            st.write('')
            st.write('')
            st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>Loading app, please wait...</p>",
                      unsafe_allow_html=True)
    
            # Mark app as ready and force rerun
            st.session_state.app_ready = True
            st.rerun()
        
        # Set up the header, login form, and check user credentials
        # Create Setup instance
        setup = Setup()
        
        # Render the header
        header = setup.header()
        
        # Render the sidebar
        sidebar = setup.navigation()
        
        # Go to login page if not yet logged in
        # if not st.session_state["authenticated"]:
            
            # Generate login form
            # user_creds = setup.login()
            
            # Pass the directory to the User class of users.py
            # user_authentication = User(self.db_file)
            
            # Pass username and password or register_new_user entries to User class of users.py.
            # Returns user_id or None for invalid creds.
            # if user_creds:
                
                # Pass username and password to Authentication class of user.py
                # user_id = user_authentication.verify_login(user_creds[0], user_creds[1])
                
                # if user_id:
                    # st.session_state["login_failed"] = False
                    # Add creds to session_state for use later
                    # st.session_state["username"] = user_creds[0]
                    # st.session_state["user_id"] = user_id
                    # st.balloons()
                    # time.sleep(2)
                    # Change authenticated to true
                    # st.session_state["authenticated"] = True
                    # Start over so session_state gets updated
                    # st.rerun()    
                    
                # else:
                    # Change login_failed to True so error message pops
                    # st.session_state["login_failed"] = True
                    # st.rerun()
        
        #---------------------------------------------------------------------------------------------
        # Make calls to import, load, and analyze data
        
        # Delete this if using the login page
        st.session_state["authenticated"] = True
        
        # Once logged in, go to linear regression page
        if st.session_state["authenticated"]:
            
            # Call Load_Data class
            load_files = Load_Data()

            # Create an upload button to load file names to be analyzed
            file_name_list = load_files.upload()
            
            st.write('')
            st.write('')
            
            # If files were uploaded
            if file_name_list:
                
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>View raw data, plots, and fit results below.</p>",
                      unsafe_allow_html=True)
                    
                st.divider()
                
                # Call the Analysis class to plot data
                plot_data = Analysis()
                 
                # Create dictionaries to store results and plot files each time we go through the loop
                # Holds raw data dataframes after user has deleted channels and/or
                # truncated the time axis using the slider bars
                edited_raw_dfs_dict = {}
                # Holds dataframes of raw and smoothed regression fit results
                all_results = {}
                # Holds plots for raw and smoothed regressions
                all_plots = {}
                # Holds rolling regression fit results
                rolling_reg_tabs = {}
                # Holds rolling regression plots
                rolling_reg_plots = {}
                
                # Loop through files
                for file in file_name_list:
                    
                    # Process the data
                    file_name, clean_filename, edited_df = self.preprocess_data(file, load_files, plot_data)
 
                    # Add the final raw edited_df to a dict for use in calculating noise below
                    # This holds data for all files selected by user for processing
                    edited_raw_dfs_dict[clean_filename] = edited_df
                    
                    # Plot the data. Pass edited_df, clean_filename, empty all_plots dict, and checkbox
                    # Returns the filename, fit results table, and all the plots (in
                    # all_plots dict)
                    f_name, results_tab, all_plots = plot_data.plot_channels(edited_df, 
                                                          clean_filename, 
                                                          all_plots)
                    
                    # Add results table and plots to dictionaries
                    # These are the final raw and smoothed data regression results
                    all_results[f_name] = results_tab
            
                    st.divider()
                    
                    # If rolling regression was checked
                    if st.session_state["rolling_regression"]:
                        
                        st.write('')
                        st.markdown(f"<p style='color: DarkRed; \
                                      font-size: 24px; \
                                      margin: 0;'>Rolling Regression Results</p>",
                                      unsafe_allow_html=True)
                        st.write('')
                        
                        # Pass in truncated df and filename
                        # Returns df of fit results and plots
                        
                        col1, col2 = st.columns([1,1])
                        
                        # Slider bar to set window size
                        with col1:
                            window_size = st.slider(
                                                    "Select rolling regression window size.",
                                                    min_value=50,
                                                    max_value=1000,
                                                    value=300,  
                                                    step=20,
                                                    # Dynamic key
                                                    key=f"win_size_{clean_filename}"  
                                                    )
                        st.write('')
                        # Instruction line
                        st.markdown(f"<p style='color: black; \
                                      font-size: 14px; \
                                      margin: 0;'>Select a channel to view plot and fit line.</p>",
                                      unsafe_allow_html=True)
                        
                        # Fit the data that falls within the window 
                        df_results = plot_data.compute_rolling_regression(edited_df, window_size, clean_filename)
                        
                        # Add to dicts for permanent storage
                        rolling_reg_tabs[clean_filename] = df_results
                        
                        # Step 3: Pass df_results to UI function for plotting
                        plot_data.rolling_reg_ui(edited_df, df_results)
                    
                        st.divider()

                # Create buttons to download all the data
                # Two options for downloading controlled by checkboxes
                # One combines all results into one file, the other displays
                # a download button for each file
                if st.session_state['download_options'] == 'Combined':
                    
                    plot_data.download_all_combined(all_results, 
                                                    all_plots, 
                                                    rolling_reg_tabs, 
                                                    rolling_reg_plots)
                    
                elif st.session_state['download_options'] == 'Separate':
                    plot_data.download_separate(all_results, 
                                                all_plots, 
                                                rolling_reg_tabs, 
                                                rolling_reg_plots)
                    
        # Noise Analysis calls
        #---------------------------------------------------------------------------------------------
                # Perform a noise analysis if Noise Analysis checkbox is checked
                if st.session_state['noise_analysis']:
                    
                    st.divider()
                    
                    st.markdown(f"<p style='color: Blue; \
                          font-size: 24px; \
                          margin: 0;'>Model Diagnostics</p>",
                          unsafe_allow_html=True)
                        
                    st.write('')
                    
                    # Call noise method to calculate noise
                    # This passes edited data, like after changing x axis or deleting a channel
                    # Returns a table of noise parameters (stdev, p value, Ljung-Box, etc)  
                    # and dictionary of residuals for raw and smoothed data (if smoothed was checked)
                    results, raw_residuals_data, smooth_residuals_data = plot_data.noise(edited_raw_dfs_dict)
                    
                    # Call classify_noise method to see if the noise is a problem
                    # as judged by certain thresholds
                    noise_class = plot_data.classify_noise(results)
                    
                    # Convert dictionaries to df
                    noise_class_df = pd.DataFrame(noise_class)
                    
                    # Make a copy for display purposes only since it will have strings
                    # that can't be used later
                    display_df = noise_class_df.copy()
                    
                    # Figure out which of the Ljung-Box p-value columns actually exist
                    # Adds "Raw LB Autocorr pval" and/or "Smooth LB Autocorr pval" to list
                    # if they are present in the df
                    autocorr_cols = [colname for colname in ["Raw LB Autocorr pval", "Smooth LB Autocorr pval"]
                                if colname in noise_class_df.columns]
                    
                    # Apply scientific notation to autocorr_cols
                    if autocorr_cols:
                        for col in autocorr_cols:
                            display_df[col] = display_df[col].apply(
                                                                    lambda x: f"{x:.2e}" if pd.notna(x) else "NA"
                                                                    )

                    # Show results in Streamlit (pretty formatting, no index)
                    st.data_editor(display_df,
                                    use_container_width=True,
                                    num_rows="static",
                                    hide_index=True
                                  )
                                        
                    # Download as csv file
                    # Convert DataFrame to CSV string
                    csv_data = noise_class_df.to_csv(index=False) 
                    
                    # Create the download button
                    st.download_button(label="Download Noise Analysis Table",
                                        data=csv_data,
                                        file_name='noise_analysis.csv',
                                        mime='text/csv'
                                        )
                    
                    st.divider()
                    
                if st.session_state['residuals_plot']:
                    
                    try:
                        # Create residuals vs. time plots
                        res_plot = plot_data.residuals_plot(pd.DataFrame(raw_residuals_data))
                        
                        # Generate the PDF independently using plotly so it can be downloaded
                        pdf_buf = plot_data.download_plots_pdf(residuals_df=pd.DataFrame(raw_residuals_data),
                                                                plot_type="residuals",
                                                                key="residuals"
                                                               ).getvalue()
     
                        # Provide download button
                        st.download_button(label="Download Residual Plots PDF",
                                            data=pdf_buf,
                                            file_name="residuals.pdf",
                                            mime="application/pdf",
                                            key="residuals_pdf"
                                          )
                        
                        st.divider()
                        
                    except:
                        st.error('Please click the Noise Calculations box. \
                                 A noise analysis must be run before the residuals can be plotted.')
                        
                if st.session_state['acf_plot']:
                    
                    try:
                        
                        # Put the residuals in a dataframe for ACF analysis
                        acf_plot_df = pd.DataFrame(raw_residuals_data)
                        residual_acf_plot = plot_data.acf_analysis(acf_plot_df) 
                        
                        # Pass to download plots function to generate pdf
                        st.download_button(label="Download ACF Plots PDF",
                                            data=plot_data.download_plots_pdf(
                                                residuals_df=pd.DataFrame(raw_residuals_data),
                                                plot_type="acf",
                                                key="acf"
                                            ).getvalue(),
                                            file_name="acf_plots.pdf",
                                            mime="application/pdf",
                                            key="acf_pdf"
                                          )
            
                        st.divider()
                        
                    except:
                        st.error('Please click the Noise Calculations box. \
                                 A noise analysis must be run before the autocorrelation function \
                                can be plotted.')
                    
                # Correct the standard error with the Newey-West and effective 
                # sample size (ESS) algorithms
                if st.session_state['acf_correction']:
                    
                    try:
                    
                        NW_ESS_corr = plot_data.residual_correction(edited_raw_dfs_dict, 
                                                                    residual_acf_plot)
                        
                        st.divider()
                        
                    except:
                        
                        st.error('Please click the Noise Calculations and ACF Plot boxes. \
                                 A noise and autocorrelation function analysis \
                                 must be run before the correction can be made.')
                        
                if st.session_state['normal']:
                    
                    try:
                        normal_check = plot_data.normal_test(pd.DataFrame(raw_residuals_data))
                        
                        st.divider()
                    
                    except:
                        st.error('Please click the Noise Calculations box. \
                                  A noise analysis must be run before the test for \
                                  normality.')   

    def preprocess_data(self, file, load_files, plot_data):

        # Get file name
        file_name = file.name
        
        # Strip out .csv from file name
        clean_filename, _ = os.path.splitext(file_name)
        
        # Import into dataframe
        import_data = load_files.import_df(file)
        
        # Grab time and oxygen concentration columns
        oxy_cols = load_files.oxy_data(import_data)
        
        # Display the raw data in an edit table
        # Returns the raw data in a dataframe with NAs dropped
        show_raw_data = plot_data.raw_data_table(oxy_cols, clean_filename)
        
        # Drop columns where all values are NaN
        raw_data_no_missing_cols = show_raw_data.apply(pd.to_numeric, errors='coerce')
        raw_data_no_missing_cols = raw_data_no_missing_cols.dropna(axis=1, how='all')
        
        # Create a multiselect box to delete unwanted channels
        # Returns file_name and data
        raw_data_dropped_channels = plot_data.delete_channels(raw_data_no_missing_cols, 
                                                              clean_filename)

        # Add a slider bar for each channel to control x axis window
        # Returns a new dataframe of time and signal for each channel
        # within the selected time window
        # These are the final raw data after the user has dropped channels and
        # truncated the time window with the slider bars
        edited_df = plot_data.slider_bars(raw_data_dropped_channels, clean_filename)

        return file_name, clean_filename, edited_df
        

        
    
        #---------------------------------------------------------------------------------------------
           
class Setup():
    """Class that lays out the app header and sidebar."""
    
    def __init__(self):
        
        pass
    
    def header(self):
        
        # Inject custom CSS from styles.py file
        st.markdown(CSS, unsafe_allow_html=True)
        
        # Draw line across the page
        st.divider()
        
        # Add a logo and title
        col1, col2 = st.columns([1,4])
        
        with col1:
        
            st.image(st.session_state['logo'])
            
        with col2:

            st.write('')
            st.write('')
            st.write('')
            # st.write('')
            
            st.markdown(f"<p style='color: Blue; \
                          font-size: 32px; \
                          margin: 0;'>Coral Health and Disease Respiration Analyzer</p>",
                          unsafe_allow_html=True
                        )
        st.divider()
        
    def navigation(self):
        
        with st.sidebar:
            
            st.write('')
            st.write('')

            st.subheader("Diagnostics")
            st.metric("Memory (MB)", f"{get_memory_usage():.2f}")
            
            col1, col2, col3 = st.columns([1,8,1])
            with col2:
                
                st.markdown(f"<p style='color: DarkBlue; \
                      font-size: 28px; \
                      margin: 0;'>Options Menu</p>",
                      unsafe_allow_html=True)
                
            st.divider()
           
            st.markdown(f"<p style='color: black; \
                  font-size: 20px; \
                  margin: 0;'><u>File Download Options<u></p>",
                  unsafe_allow_html=True)
                
            st.write('')
                
            col1, col2, col3 = st.columns([0.15,8,1])
            with col2:
                    
                download_options = st.radio(label = 'File download options.', 
                                            options = ['Combined', 'Separate'], 
                                            width=800,
                                            label_visibility="collapsed")
            st.write('')
            
            st.session_state['download_options'] = download_options
            
            st.divider()
            
            st.markdown(f"<p style='color: black; \
                  font-size: 20px; \
                  margin: 0;'><u>Smooth Data<u></p>",
                  unsafe_allow_html=True)
                    
            st.write('')
            
            col1, col2, col3 = st.columns([0.15,8,1])
            with col2:
                    
                st.session_state.smooth_data = st.checkbox('SavGol Filter', width = 800)
                
            st.divider()    
            
            st.markdown(f"<p style='color: black; \
                  font-size: 20px; \
                  margin: 0;'><u>Model Diagnostics<u></p>",
                  unsafe_allow_html=True)
                
            st.write('')
            
            col1, col2, col3 = st.columns([0.15,8,1])
            with col2:
            
                # Create checkboxes in the session state for model diagnostics 
                st.session_state.noise_analysis = st.checkbox('Noise Calculations', width = 800)
                st.session_state.residuals_plot = st.checkbox('Residuals Plot', width = 800)
                st.session_state.acf_plot = st.checkbox('ACF Plot', width = 800)
                st.session_state.normal = st.checkbox('Normal Distribution Tests', width = 800)
            
            st.divider()
        
            # Checkbox for autocorrelation correction
            st.markdown(f"<p style='color: black; \
                  font-size: 20px; \
                  margin: 0;'><u>Signal Processing<u></p>",
                  unsafe_allow_html=True)
                
            st.write('')
            
            col1, col2, col3 = st.columns([0.15,8,1])
            with col2:
                
                # Perform rolling regression
                st.session_state.rolling_regression = st.checkbox('Rolling Regression', width = 800)
            
                # Correct using Newey-West and ESS
                st.session_state.acf_correction = st.checkbox('Correlation Correction', width = 800)
            
            st.divider()
            
            # Box with definitions of terms for the user to access
            # to learn about what is being calculated in the app
            # st.markdown(f"<p style='color: black; \
            #       font-size: 20px; \
            #       margin: 0;'><u>About<u></p>",
            #       unsafe_allow_html=True)
                
            st.write('')
            
            with st.expander('Learn about the Respiration Analyzer 🪸'):
                
                st.markdown(
                            """
                            ### 📂 File download options
                            - **Combined** → generates one **CSV** file and one **PDF** with all results  
                            - **Separate** → generates separate files for all results  
                            
                            ---
                            
                            ### 📊 SavGol Filter
                            Savitsky–Golay smoothing fits a polynomial to a sliding window of points.  
                            - Window size: **50 points**  
                            - Polynomial order: **3rd**  
                            
                            ---
                            
                            ### 🎛️ Noise Analysis
                            - **LB Autocorr pval** → Ljung-Box algorithm calculates correlation between residuals.
                            Residual correlation doesn't affect the slope but underestimates the standard error.
                            It also indicates a delayed response somewhere in the experiment. LB p-value is > 0.005
                            for no correlation and < 0.005 for correlation (95% CI).'
                            - **Ransac** → RANdom SAmple Consensus determines the number of outliers in the time 
                            series. A large % could indicate noise in the experiment or the data do not conform
                            to the model well. 
                            - **Residuals Plots** → Residuals should be scattered randomly around the 0 line
                            Any non-random pattern in the residuals indicates autocorrlation or heteroskedasticity.
                            This violates one of the assumptions of linear regression and indicates sluggishness
                            in the experiment (slow mixing, sensor response) and yields underestimates in standard
                            errors, confidence intervals, and p-values. These can be corrected (see below).
                            - **Autocorrelation Function Plots** → Shows the strength and direction of the 
                            correlation from -1 to 1 with 95% CI. The x axis represents the correlation of
                            a data point with other points n "lags" or time periods preceding that point. Any
                            bar above the 95% CI regions are statistically autocorrelated.
                            - **Normality Tests** → The histogram of residuals should be normally distributed.
                            This is another fundamental assumption for linear regression. The Q-Q plot (quantile-quantile)
                            should be linear.
                            
                            ---
                            
                            ### 📈 Signal Processing
                            - **Rolling Regression** → Performs a linear regression across the selected time series
                            data in windows set by the slider bar and in steps of 5% of the slider bar setting. 
                            The regressions are ranked by R2 and the top 10 regressions are displayed in a table. 
                            - **Correlation Correction** → When residuals are correlated, the usual error bars 
                            and p-values become too optimistic. The Newey-West method corrects this by widening 
                            the error bars so they better reflect reality, and it should be used when reporting 
                            slopes, confidence intervals, and p-values. The Effective Sample Size (ESS) method 
                            takes a different approach: it reduces the number of “independent” data points to 
                            account for correlation, then recalculates the error bars and p-values. ESS is best 
                            treated as a diagnostic — a small ESS suggests sluggish or delayed responses in 
                            the experiment — while Newey-West provides the corrected statistics you should 
                            use for inference.
                            """,
                            unsafe_allow_html=True
                            )  
                
                    
            st.divider()
                        
    def login(self, login_failed=False):
        """Handles login authentication."""
        
        st.write('')
        
        with st.form('login_form', border = False, enter_to_submit=False):

            # Input box widgets and button
            self.username = st.text_input("Username", key="login_username")
            self.password = st.text_input("Password", type="password", key="password")
            self.login = st.form_submit_button("Login")
            
            # Show error if session flag is set
            if st.session_state.get("login_failed"):
                st.error("Invalid username or password. Please try again.")
            
            if self.login:
                if not self.username or not self.password:
                    
                    st.error("Please fill in all fields.")
                    
                else:
                    return self.username, self.password
                    self.login_placeholder = st.empty()
       

class Load_Data():
    """Creates a file uploader and imports data as dataframe."""
    
    def __init__(self):
        
        pass
    
    def upload(self):
        
        # File uploader that allows multiple files
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=["csv", "txt"],  
            accept_multiple_files=True
            )
        
        return uploaded_files
        
    @staticmethod    
    @st.cache_data
    def import_df(file):
        
        """
        Import a .txt or .csv file into a pandas DataFrame.
        - TXT: assumes tab-delimited, skips metadata rows.
        - CSV: tries multiple common encodings.
        Returns DataFrame on success, None on failure.
        """
        # Handle TXT files
        if file.name.endswith(".txt"):
            try:
                file.seek(0)
                df = pd.read_csv(file, sep="\t", encoding='latin1', skiprows=18)
                return df
            except Exception as e:
                st.error(f"❌ Failed to read TXT file {file.name}: {e}")
                return None
    
        # Handle CSV files
        elif file.name.endswith(".csv"):
            encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            for enc in encodings_to_try:
                try:
                    # file.seek(0)
                    df = pd.read_csv(file, encoding=enc, skiprows=18)
                    return df
                
                except UnicodeDecodeError:
                    continue
                
                except Exception as e:
                    st.error(f"❌ Error reading CSV file {file.name}: {e}")
                    return None
                
            st.error(f"❌ Unable to read CSV file {file.name}. Please save as UTF-8 CSV.")
            return None
    
        else:
            st.error(f"❌ Unsupported file type: {file.name}")
            return None

        # If it's a CSV file, try robust CSV import
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file, encoding=enc)
                return df
                
            except UnicodeDecodeError:
                file.seek(0)
                continue

        st.error(
                f"❌ Unable to read {file.name}. "
                "Please re-save your file as UTF-8 CSV."
                )
        return None
        
    @staticmethod
    @st.cache_data
    def oxy_data(df):
        """Retrieves oxygen channels and time."""
        
        oxy_df = df.iloc[:, [2, 4, 5, 6, 7]]
        oxy_df.columns = ['time (s)', 'Ch1', 'Ch2', 'Ch3', 'Ch4']

        return oxy_df
    
    
class Analysis():
    
    """Performs smoothing, plotting, linear regression, noise, 
    and normal distribution analysis functions."""
    
    def __init__(self):
        
        pass
    
    def raw_data_table(self, oxy_df, file_name):
        
        st.write('')
        st.write('')
        
        # Write the filename
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>{file_name}</p>",
              unsafe_allow_html=True)
        
        st.write('')
            
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Raw Data</p>",
              unsafe_allow_html=True)
    
        # Create a data table 
        raw_data_df = st.data_editor(oxy_df, 
                                   use_container_width=True,
                                   num_rows="static",
                                   hide_index=True,
                                   key=f"raw{file_name}"
                                   )
        
        # Drop rows with NaN 
        naScrubbed_raw_data_df = raw_data_df.dropna(subset=['time (s)', 'Ch1', 'Ch2', 'Ch3', 'Ch4'])

        del raw_data_df
        gc.collect()
        
        return naScrubbed_raw_data_df
        
    def delete_channels(self, raw_data_no_missing_cols, file_name):
        
        st.write('')
        
        st.markdown(f"<p style='color: DarkBlue; \
                      font-size: 18px; \
                      margin: 0; \
                      font-style: italic;'>Delete channels.</p>",
                      unsafe_allow_html=True)
            
        st.write('')
        
        # Add multiselect to drop columns
        # Create list with all channels, leaving out time
        all_channels = [c for c in raw_data_no_missing_cols.columns if c.startswith("Ch")]
        
        # Create multiselect box
        drop_cols = st.multiselect("Delete channels.", 
                                   options=all_channels, 
                                   placeholder='Choose channels to drop.',
                                   width = 300, 
                                   label_visibility = "collapsed",
                                   key=f"drop_{file_name}")
        
        # Dropped selected channels from the table
        edited_df = raw_data_no_missing_cols.drop(columns=drop_cols, errors="ignore")
    
        # Create dynamic list of channels to use for analysis
        channels = [c for c in all_channels if c not in drop_cols]
  
        # If all channels are accidentally deleted, throw an error and return empty 
        if not channels:
            st.error("No channels selected for analysis.")
            return file_name, pd.DataFrame()
        
        # Take time column and the channel cols that weren't deleted,
        # convert columns to numeric, and coerce errors to NaN.
        edited_df[['time (s)'] + channels] = edited_df[['time (s)'] + 
                                                       channels].apply(pd.to_numeric, 
                                                                       errors="coerce"
                                                                      )
                                                                       
        return edited_df
                                                         
    
    def slider_bars(self, raw_data_dropped_channels, file_name):
        
        """
        Adds independent sliders for each channel in the dataframe
        and returns a new dataframe where each channel is masked to
        its own time window.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'time (s)' column plus one or more channel columns.
        file_name : str
            Unique identifier for Streamlit widget keys.
        
        Returns
        -------
        pd.DataFrame
            A copy of df where each channel is masked to its selected
            time window. Values outside the window are set to NaN.
        """
        
        st.write('')
        
        st.markdown(f"<p style='color: DarkBlue; \
                      font-size: 18px; \
                      margin: 0; \
                      font-style: italic;'>Select a time window for each channel.</p>",
                      unsafe_allow_html=True)
            
        st.write('')
        
        # Copy to avoid modifying original
        edited_df = raw_data_dropped_channels.copy()
        
        # Get the time data
        time = edited_df['time (s)']
        
        # Get existing channel cols since some may be deleted
        channel_cols = [c for c in edited_df.columns if c != 'time (s)']
    
        # Loop through channels two at a time to build side-by-side (2 col) slider bars
        for i in range(0, len(channel_cols), 2):
            
            # Create three columns, one for a little space: left slider, spacer, right slider
            col1, spacer, col2 = st.columns([1, 0.1, 1])
            
            # Get index and col name from channel cols in groups of 2
            for j, col in enumerate(channel_cols[i:i+2]):
                target_col = col1 if j == 0 else col2
                with target_col:
                    # Print the channel number
                    st.write(f"**{col}**")
                    # Create slider bar
                    t_min, t_max = st.slider(
                                            f"Select time window for {col}",
                                            min_value=float(time.min()),
                                            max_value=float(time.max()),
                                            value=(float(time.min()), float(time.max())),
                                            step=1.0,
                                            label_visibility = "collapsed",
                                            key=f"time_slider_{file_name}_{col}"
                                            )
                    # Create a mask that includes the times the user wants to keep
                    mask = (time >= t_min) & (time <= t_max)
                    # Set all times outside the mask to nan
                    edited_df.loc[~mask, col] = float('nan')
     
        return edited_df
     
    def plot_channels(self, edited_df, file_name, all_plots):
        """
        Plot all channels in edited_df with optional Savgol filtering.
        Automatically handles NaNs from slider selection.
    
        Parameters
        ----------
        edited_df : pd.DataFrame
            Must contain a "time (s)" column and one or more channel columns.
        file_name : str
            Name of file for labeling/saving.
        all_plots : dict
            Dictionary to hold generated figures.
        apply_savgol : bool
            Whether to apply Savitzky–Golay filter.
        """
        
        # Grab time values
        time = edited_df["time (s)"].values
        
        # Create a list for regression results dict
        results = []

        # Make sure any existing plots are closed
        plt.close("all")
        
        # Create plot instance size
        figsize = (6, 4)
    
        st.write("")
    
        # If smoothing was checked by user, set up two cols for side-by-side
        # raw and smoothed data plots
        if st.session_state['smooth_data']:
            col1, col2 = st.columns(2)
            
        else:
            
            # Create container for raw data plot
            col1 = st.container()
            col2 = None
    
        # Raw data plot
        with col1:
            
            fig_raw, ax = plt.subplots(figsize=figsize)
    
            # Loop through channels and process y data
            for ch in edited_df.drop(columns=["time (s)"]).columns:
                
                # Grab y values
                y = edited_df[ch].values
    
                # Mask NaNs (deleted points via sliders)
                mask = ~np.isnan(y)
                # These are the x and y values the user wanted to keep
                # according to the position of the slider bar
                x_valid = time[mask]
                y_valid = y[mask]
                
                # Get low and high times to do a time range column in results
                low_time = x_valid.min()
                high_time = x_valid.max()
    
                # Plot only valid points
                ax.plot(x_valid, y_valid, label=ch)
    
                # Call fit_regression function 
                slope, intercept, r, p, stderr, slope_ci = self.fit_regression(x_valid, y_valid)
                
                # In case slope couldn't be calculated
                if slope != "NA":
                    # Add linear trend line
                    ax.plot(x_valid, slope * x_valid + intercept, "--")
                
                # Append results to list as a dict
                results.append({
                                "File": file_name,
                                "Channel": ch,
                                "Type": "Raw",
                                "Time Window": f"{low_time} - {high_time}",
                                "Slope": slope,
                                "Slope 95% CI": slope_ci,
                                "Intercept": round(intercept,1),
                                "R2": round(r * r if r != "NA" else "NA",3),
                                "p": f"{p:.2e}" if p != "NA" else "NA",
                                "slope std error": stderr
                               })
            
            # Plot labels and legends
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal")
            ax.set_title(f"Raw: {file_name}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig_raw, clear_figure=False)
            all_plots[f"{file_name}_raw"] = fig_raw
    
            # Save y-limits for filtered plot
            raw_ylim = ax.get_ylim()
            
            # Close the plots once rendered to avoid memory leak
            plt.close(fig_raw)
            del fig_raw, ax         
            gc.collect()
    
        # Filtered data plots; Same strategy but rendered side-by-side with raw data plot
        if st.session_state['smooth_data']:
            
            with col2:
                
                fig_filtered, ax = plt.subplots(figsize=figsize)
    
                for ch in edited_df.drop(columns=["time (s)"]).columns:
                    y = edited_df[ch].values
    
                    # Mask NaNs
                    mask = ~np.isnan(y)
                    x_valid = time[mask]
                    y_valid = y[mask]
                    
                    # Get low and high times to do a time range column in results
                    low_time = x_valid.min()
                    high_time = x_valid.max()
    
                    # Savgol filter 
                    if len(y_valid) > 3:
                        window_length = min(51, len(y_valid) // 2 * 2 + 1)
                        y_filtered = savgol_filter(y_valid, window_length=window_length, polyorder=3)
                    
                    else:
                        y_filtered = y_valid.copy()
   
                    # Plot filtered data
                    ax.plot(x_valid, y_filtered, label=f"{ch} filtered")
    
                    # Linear regression on filtered data using the fit regression function below
                    slope, intercept, r, p, stderr, slope_ci = self.fit_regression(x_valid, y_filtered)
                    if slope != "NA":
                        ax.plot(x_valid, slope * x_valid + intercept, "--")
    
                    results.append({
                                    "File": file_name,
                                    "Channel": ch,
                                    "Type": "Filtered",
                                    "Time Window": f"{low_time} - {high_time}",
                                    "Slope": slope,
                                    "Slope 95% CI": slope_ci,
                                    "Intercept": round(intercept,1),
                                    "R2": round(r * r if r != "NA" else "NA",3),
                                    "p": f"{p:.2e}" if p != "NA" else "NA",
                                    "slope std error": stderr
                                   })
    
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Signal")
                ax.set_title(f"Filtered: {file_name}")
                ax.set_ylim(raw_ylim)  # match y-axis with raw plot
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_filtered, clear_figure=False)
                all_plots[f"{file_name}_filtered"] = fig_filtered

                plt.close(fig_filtered)
                del fig_filtered, ax
                gc.collect()
    
        # Regression results
        # Create dataframe from results dict
        results_df = pd.DataFrame(results)
        
        st.markdown(
                    "<p style='color: DarkRed; font-size: 24px; margin: 0;'>Linear Regression Results</p>",
                    unsafe_allow_html=True
                    )
    
        # Show results
        results_table = st.data_editor(
                                        results_df,
                                        use_container_width=True,
                                        num_rows="static",
                                        hide_index=True,
                                        key=f"{file_name}"
                                       )
    
        return file_name, results_table, all_plots
        
    def fit_regression(self, x, y):
        """
        Run linear regression and return slope, intercept, r, p, stderr, slope_ci.
        If insufficient points, returns "NA".
        Called from plot_channels()
        """
        # If time axis doesn't have enough points for a fit, make all fit params NA
        if len(x) < 2:
            return ("NA",) * 6
    
        # Regress
        slope, intercept, r, p, stderr = linregress(x, y)
        
        # Get the number of points in the dataset
        n = len(x)
        
        # Calculate the degrees of freedom. Make sure this is at least 0.
        df_resid = max(0, n - 2)
        if df_resid > 0:
            # Use scipy t.ppf to calculate the student's t score 
            t_val = t.ppf(1 - 0.025, df_resid)
            # Calculate low and high CI for 95% CI
            ci_low = slope - t_val * stderr
            ci_high = slope + t_val * stderr
            slope_ci = f"{ci_low:.6f} - {ci_high:.6f}"
        
        else:
            slope_ci = "NA"
    
        return slope, intercept, r, p, stderr, slope_ci

    def noise(self, all_raw_data, lb_lags=20, lb_alpha=0.05):
        """
        Perform noise analysis on each file and channel in all_raw_data.
        Accepts a dictionary of residuals.
        Returns a dataFrame with noise metrics and accept/reject column.
        Also returns the residuals.
        If smooth=True, also run Savitzky–Golay smoothing and analyze.
        """
        
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Noise Analysis Results</p>",
              unsafe_allow_html=True)
            
        # Create results lists
        results = []
        raw_residuals_data = []
        smooth_residuals_data = []  
    
        for file_name, df in all_raw_data.items():
            
            # Reshape time for use with scikit learn
            t = df['time (s)'].values.reshape(-1, 1)
    
            # Get y values from channel columns
            for ch in [c for c in df.columns if c.startswith("Ch")]:
        
                # Convert to numeric
                y_raw = pd.to_numeric(df[ch], errors="coerce").values
        
                # Drop NaNs from the np array
                mask = ~np.isnan(y_raw)
                
                # Get time and y data without NaNs
                t_clean = t[mask]
                y_clean = y_raw[mask]
                
                # Gotta have data
                if len(y_clean) == 0 or len(t_clean) == 0:
                    continue
    
                # Raw data residuals
                lr = LinearRegression().fit(t_clean, y_clean)
                residuals_raw = y_clean - lr.predict(t_clean)
                raw_residual_std = np.std(residuals_raw)
                raw_percent_residual_std = 100 * (raw_residual_std / np.mean(y_clean))
    
                # Ljung-Box autocorrelation
                if len(residuals_raw) > lb_lags:
                    try:
                        lb_raw = acorr_ljungbox(residuals_raw, lags=[lb_lags], return_df=True)
                        lb_pval_raw = float(lb_raw["lb_pvalue"].iloc[0])
                        
                    except Exception as e:
                        st.warning(f"Ljung-Box failed on {file_name}, {ch}: {e}")
                        lb_pval_raw = np.nan
                else:
                    lb_pval_raw = np.nan
               
                # Use random sample consensus (ransac) to find outliers and inliers
                # Fit line to the data while ignoring outliers
                # Need to set the seed because calculations are slightly different every time
                ransac_raw = RANSACRegressor(random_state=42).fit(t_clean, y_clean)
                
                # Create a boolean array where True are inliers
                inlier_mask_raw = ransac_raw.inlier_mask_
                # Calculate standard deviation of the outliers; this is a measure of how the noisy 
                # the data are. If there are no outliers, return 0
                # ransac_noise_raw = np.std(y_clean[~inlier_mask_raw]) if np.any(~inlier_mask_raw) else 0.0
                
                # Fraction of data points that are outliers
                raw_outlier_fraction = np.mean(~inlier_mask_raw)
                
                # Standard deviation of inliers (typical scatter/noise)
                # raw_inlier_std = np.std(y_clean[inlier_mask_raw]) if np.any(inlier_mask_raw) else 0.0
    
                # Base result always includes raw
                result_dict = {
                                "File": file_name,
                                "Channel": ch,
                                "Raw % Residual STD": round(raw_percent_residual_std,2),
                                "Raw LB Autocorr pval": lb_pval_raw, 
                                # "Raw Ransac": round(ransac_noise_raw,3),
                                "Raw % Outliers": round(100*raw_outlier_fraction,2),
                                # "Raw Inlier Std":round(raw_inlier_std,3)
                              }
    
                # Optional: noise analysis on smoothed data if user selected checkbox
                if st.session_state['smooth_data']:
                    # Smooth Data (Savitzky–Golay) and repeat noise analysis
                    window_length = min(51, len(y_clean) // 2 * 2 + 1)
                    polyorder = 3 if window_length > 3 else 2
    
                    y_smooth = savgol_filter(y_clean, window_length=window_length, polyorder=polyorder)
            
                    lr_smooth = LinearRegression().fit(t_clean, y_smooth)
                    residuals_smooth = y_smooth - lr_smooth.predict(t_clean)
                    smooth_residual_std = np.std(residuals_smooth)
                    smooth_percent_residual_std = 100 * (smooth_residual_std / np.mean(y_smooth))
    
                    if len(residuals_smooth) > lb_lags:
                        try:
                            lb_smooth = acorr_ljungbox(residuals_smooth, lags=[lb_lags], return_df=True)
                            lb_pval_smooth = float(lb_smooth["lb_pvalue"].iloc[0])
                            
                        except Exception as e:
                            st.warning(f"Ljung-Box failed on {file_name}, {ch} (smooth): {e}")
                            lb_pval_smooth = np.nan
                    else:
                        lb_pval_smooth = np.nan
                        
                    # Ransac on smoothed data
                    ransac_smooth = RANSACRegressor(random_state=42).fit(t_clean, y_smooth)
                    inlier_mask_smooth = ransac_smooth.inlier_mask_
                    # ransac_noise_smooth = np.std(y_smooth[~inlier_mask_smooth]) if np.any(~inlier_mask_smooth) else 0.0
                    
                    # Fraction of data points that are outliers
                    smooth_outlier_fraction = np.mean(~inlier_mask_smooth)
                    
                    # Standard deviation of inliers (typical scatter/noise)
                    # smooth_inlier_std = np.std(y_clean[inlier_mask_raw]) if np.any(inlier_mask_raw) else 0.0
                    
                    # Add smooth metrics to the same row
                    result_dict.update({
                                        "Smooth % Residual STD": round(smooth_percent_residual_std, 2),
                                        "Smooth LB Autocorr pval": lb_pval_smooth,
                                        "Smooth % Outliers": round(100*smooth_outlier_fraction,2),
                                       })
    
                    # Append smooth residuals for plotting later
                    smooth_residuals_data.append({
                                                "File": file_name,
                                                "Channel": ch,
                                                "time": t_clean.flatten(),
                                                "residuals": residuals_smooth
                                                })
    
                results.append(result_dict)
                
                # Append raw residuals and time to raw_residuals_data list to plot later
                # Store time + residuals
                raw_residuals_data.append({
                                            "File": file_name,
                                            "Channel": ch,
                                            "time": t_clean.flatten(),
                                            "residuals": residuals_raw,
                                          })
    
        return results, raw_residuals_data, smooth_residuals_data
    
    def classify_noise(self, 
                       results,
                       percent_resid_std_thresh=15,
                       lb_alpha=0.05,
                       ransac_thresh=25):
        """
        Classify Raw and (if available) Smooth noise as Accept or Warning.
        """
    
        for row in results:
            # Always classify raw data
            # List of criteria for Raw data
            raw_flags = [
                        row["Raw % Residual STD"] > percent_resid_std_thresh,
                        row["Raw LB Autocorr pval"] < lb_alpha,
                        row["Raw % Outliers"] > ransac_thresh
                        ]
            
            row["Raw Checks"] = "Warning" if any(raw_flags) else "Accept"
    
            # Conditionally classify smoothed data
            if st.session_state['smooth_data']:
                
                smooth_flags = [
                                row["Smooth % Residual STD"] > percent_resid_std_thresh,
                                row["Smooth LB Autocorr pval"] < lb_alpha,
                                row["Smooth % Outliers"] > ransac_thresh
                               ]
    
                row["Smooth Checks"] = "Warning" if any(smooth_flags) else "Accept"

    
        return results

    def residuals_plot(self, residuals_df):
        """
        Generates residuals plots.
        Accepts a dataframe of residuals for each channel.
        """
        
        st.write('')
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Residuals Plots</p>",
              unsafe_allow_html=True)
        
        # Flatten first
        rows = []
        for _, row in residuals_df.iterrows():
            file = row["File"]
            ch = row["Channel"]
            times = row["time"]
            resids = row["residuals"]
    
            for t, r in zip(times, resids):
                rows.append({"File": file, "Channel": ch, "time": t, "residuals": r})
    
        residuals_df_flat = pd.DataFrame(rows)
    
        # Two-column layout
        col1, col2 = st.columns(2)
        cols = [col1, col2]
    
        # Group by File & Channel
        groups = list(residuals_df_flat.groupby(["File", "Channel"]))
    
        # Store figures for download
        figs = {}
    
        for i, ((file, ch), group) in enumerate(groups):
            # Create Plotly figure
            fig = go.Figure()
    
            # Scatter residuals
            fig.add_trace(go.Scatter(
                x=group["time"],
                y=group["residuals"],
                mode="markers",
                marker=dict(size=5, color="steelblue", opacity=0.6),
                name=f"{file} - {ch}"
            ))
    
            # Zero line
            fig.add_hline(y=0, line_color="red", line_dash="dash", line_width=1)
    
            # Layout styling
            fig.update_layout(
                title=dict(text=f"{file} - {ch}", x=0.5, xanchor="center"),  # centered title
                xaxis=dict(
                    title="Time (s)",
                    titlefont=dict(size=16, color="blue"),
                    tickfont=dict(size=12, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                yaxis=dict(
                    title="Residuals",
                    titlefont=dict(size=16, color="blue"),
                    tickfont=dict(size=12, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                margin=dict(l=60, r=30, t=40, b=50),
                width=450,
                height=300
            )
    
            # Alternate between left/right column
            cols[i % 2].plotly_chart(fig, use_container_width=True)
    
            # Save for download
            figs[f"{file}_{ch}"] = fig
        
    def acf_analysis(self, residuals_df):
        """
        Plot autocorrelation function of residuals for each file/channel.
        Calculates maxlag and returns df of file, channel, and maxlag.
        Includes 95% confidence intervals like matplotlib's plot_acf.
        """
        
        st.write('')
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Residual Autocorrelation Function Plots</p>",
              unsafe_allow_html=True)

        st.write('')

        # Adjust lags from 10 to 200 with a default of 50
        col1, col2 = st.columns([1,1])
        with col1:
            lags = st.slider(
                            label="Number of lags to display in ACF plot",
                            min_value=10,
                            max_value=200,
                            value=50, 
                            step=1
                            )
        
        # Make two columns
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        
        maxlag_list = []
    
        for i, (_, row) in enumerate(residuals_df.iterrows()):
            file_name = row["File"]
            channel = row["Channel"]
            residuals = np.array(row["residuals"])
    
            # Compute ACF values
            acf_vals = acf(residuals, nlags=lags, fft=True)
    
            # Bartlett CI bands (same as plot_acf default)
            N = len(residuals)
            ci = 1.96 / np.sqrt(N)
            lags_range = np.arange(len(acf_vals))
    
            # Create Plotly figure
            fig = go.Figure()
    
            # Bar plot of ACF
            fig.add_trace(go.Bar(
                x=lags_range,
                y=acf_vals,
                marker_color="steelblue",
                name="ACF"
            ))
    
            # Add CI bands
            fig.add_hline(y=ci, line_dash="dot", line_color="red", annotation_text="+95% CI")
            fig.add_hline(y=-ci, line_dash="dot", line_color="red", annotation_text="-95% CI")
    
            # Layout
            fig.update_layout(
                title=dict(
                    text=f"{file_name} - {channel}",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color="blue")
                ),
                xaxis=dict(
                    title="Lag",
                    titlefont=dict(size=16, color="blue"),
                    tickfont=dict(size=12, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                yaxis=dict(
                    title="Autocorrelation",
                    titlefont=dict(size=16, color="blue"),
                    tickfont=dict(size=12, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                margin=dict(l=60, r=30, t=50, b=50),
                width=450,
                height=300
            )
    
            # Alternate between col1 and col2
            cols[i % 2].plotly_chart(fig, use_container_width=True)

    
            # Estimate maxlag
            maxlag = self.estimate_maxlags(residuals)
            maxlag_list.append({
                "File": file_name,
                "Channel": channel,
                "Maxlag": maxlag
            })
    
            del fig
            gc.collect()
        
        maxlag_df = pd.DataFrame(maxlag_list)
        return maxlag_df
                
    def estimate_maxlags(self, residuals, max_lag=200):
        """
        Estimate maxlags by finding the first lag where ACF ~ 0.
        If no crossing, fallback to max_lag.
        """
        acf_vals = acf(residuals, nlags=max_lag, fft=True)
        N = len(residuals)
        threshold = 2 / np.sqrt(N)  # white noise CI
        
        for lag in range(1, len(acf_vals)):
            if abs(acf_vals[lag]) < threshold:
                return lag
            
        # fallback position
        return max_lag  
        
    def residual_correction(self, edited_raw_dfs_dict, maxlag_df): 
        """
        Apply Newey-West and ESS corrections to all files and channels.
        Returns a Streamlit editable table with slope, NW SE, ESS SE, 95% CI, intercept, p-value, Maxlag, and Effective N.
        """
        
        results = []
    
        for file_name, df in edited_raw_dfs_dict.items():
            
            time = df['time (s)'].values
            
            # Only keep columns that start with "Ch" AND have at least one non-NA value
            channels = [c for c in df.columns if c.startswith("Ch") and df[c].notna().any()]
    
            for ch in channels:
                
                # Drop NaNs from channel and corresponding time
                y_raw = df[ch].values
                mask = ~np.isnan(y_raw)
                y = y_raw[mask]
                t_clean = time[mask]
    
                # Lookup maxlag for this file/channel
                row = maxlag_df[
                                (maxlag_df["File"] == file_name) &
                                (maxlag_df["Channel"] == ch)
                                ]
                
                lags = int(row["Maxlag"].values[0])
    
                # OLS for residuals
                # Make a 2D array for sm.OLS fit
                X = sm.add_constant(t_clean)
                ols_model = sm.OLS(y, X).fit()
                residuals = ols_model.resid
                N = len(residuals)
    
                # Newey-West correction
                nw_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
                # Get slope, intercept, standard error, and p value
                slope = nw_model.params[1]
                intercept = nw_model.params[0]
                nw_se = nw_model.bse[1]
                NW_slope_pval = nw_model.pvalues[1]
    
                # 95% CI for NW SE
                ci_low_nw = slope - 1.96 * nw_se
                ci_high_nw = slope + 1.96 * nw_se
                slope_ci_nw = f"{ci_low_nw:.6f} – {ci_high_nw:.6f}"
    
                # --- ESS correction to Newey-West ---
                if N > 3:
                    # Make sure lags ≤ N-1
                    lags = min(lags, N - 1)
    
                    acf_vals = acf(residuals, nlags=lags, fft=True)
                    # Exclude lag 0
                    rho_sum = np.sum(acf_vals[1:lags+1])  
                    Neff = N / (1 + 2 * rho_sum)
    
                    ess_se = nw_se * np.sqrt(N / Neff) if Neff > 0 else np.nan
    
                    # 95% CI for ESS SE
                    if not np.isnan(ess_se):
                        ci_low_ess = slope - 1.96 * ess_se
                        ci_high_ess = slope + 1.96 * ess_se
                        slope_ci_ess = f"{ci_low_ess:.6f} - {ci_high_ess:.6f}"
    
                        # p value for ESS
                        df_ess = max(Neff - 2, 1)  # avoid invalid df
                        t_stat_ess = slope / ess_se
                        ESS_slope_pval = 2 * (1 - stats.t.cdf(abs(t_stat_ess), df=df_ess))
                    else:
                        slope_ci_ess = np.nan
                        ESS_slope_pval = np.nan
                else:
                    Neff = np.nan
                    ess_se = np.nan
                    slope_ci_ess = np.nan
                    ESS_slope_pval = np.nan
                # -----------------------------------
    
                # Append results
                results.append({
                    "File": file_name,
                    "Channel": ch,
                    "Intercept": intercept,
                    "Slope": slope,
                    "NW SE": nw_se,
                    "NW 95% CI": slope_ci_nw,
                    "NW p-value": f"{NW_slope_pval:.2e}",
                    "ESS SE": ess_se,
                    "ESS 95% CI": slope_ci_ess if slope_ci_ess is not np.nan else "NA",
                    "ESS p-value": f"{ESS_slope_pval:.2e}" if not np.isnan(ESS_slope_pval) else "NA",
                    "Maxlag used": lags,
                    "Effective N": int(Neff) if not np.isnan(Neff) else "NA"
                })
    
        # Convert to DataFrame and display in Streamlit
        results_df = pd.DataFrame(results)
    
        st.write('')
        st.markdown("<p style='color: DarkRed; font-size: 24px; margin:0;'>Linear Regression Results Corrected for Autocorrelation</p>",
                    unsafe_allow_html=True)
        
        st.write('')
    
        corrected_results_df = st.data_editor(results_df, 
                                              use_container_width=True, 
                                              num_rows="dynamic", 
                                              hide_index=True)
    
        # Download button
        csv_data = results_df.to_csv(index=False)
        st.download_button(label="Download Corrected Results",
                           data=csv_data,
                           file_name='correlation_corrected.csv',
                           mime='text/csv')
    
        return corrected_results_df
        
    def normal_test(self, residuals_df):
        
        """
        Plot histogram + Q-Q plot of residuals for each file/channel.
        Assumes residuals_df has columns: ["File", "Channel", "residuals"]
        where residuals is a sequence (list, np.array, or pd.Series).
        """
     
        st.write('')
        st.markdown(f"<p style='color: DarkRed; \
              font-size: 24px; \
              margin: 0;'>Residual Normality Checks</p>",
              unsafe_allow_html=True)
     
        # Two columns
        col1, col2 = st.columns(2)
        cols = [col1, col2]

        plt.close("all")
     
        for i, (_, row) in enumerate(residuals_df.iterrows()):
            file_name = row["File"]
            channel = row["Channel"]
            residuals = np.array(row["residuals"]).astype(float)
     
            # Make side-by-side subplots: Histogram + Q-Q
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
     
            # Histogram
            axes[0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
            axes[0].set_title("Histogram")
            axes[0].set_xlabel("Residuals")
            axes[0].set_ylabel("Frequency")
     
            # Q–Q plot
            q_q = stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title("Q–Q Plot")
     
            fig.suptitle(f"{file_name} - {channel}")
     
            # Alternate between col1 and col2
            cols[i % 2].pyplot(fig)
            
            plt.close(fig)
            del fig
            gc.collect()
                
    # @staticmethod
    # @st.cache_data
    def compute_rolling_regression(self, edited_df, window_size, file_name="file", step_perc=5):
        results = []
        time = edited_df.iloc[:, 0].values
        step = int(window_size * step_perc * 0.1)
        
        for channel in edited_df.columns[1:]:
            signal = edited_df[channel].values
            heap = []  # per-channel heap for top 4 windows
        
            for start in range(0, len(time) - window_size, step):
                end = start + window_size
                x = time[start:end]
                y = signal[start:end]
        
                mask = ~np.isnan(y)
                x, y = x[mask], y[mask]
                if len(x) == 0:
                    continue
        
                slope, intercept, r, p, stderr, slope_ci = self.fit_regression(x, y)
                if slope == "NA":
                    continue
        
                # Newey-West correction
                X = sm.add_constant(x)
                ols_model = sm.OLS(y, X).fit()
                residuals = ols_model.resid
                r2 = ols_model.rsquared
                lags = self.estimate_maxlags(residuals, max_lag=200)
                nw_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
                nw_slope = nw_model.params[1]
                nw_intercept = nw_model.params[0]
                nw_se = nw_model.bse[1]
                nw_pval = nw_model.pvalues[1]
                ci_low_nw = nw_slope - 1.96 * nw_se
                ci_high_nw = nw_slope + 1.96 * nw_se
        
                window_dict = {
                    "Channel": channel,
                    "Window": f"{x[0]:.1f} – {x[-1]:.1f}",
                    "Start": x[0],
                    "End": x[-1],
                    "Slope": round(slope, 4),
                    "Slope Stderr": round(stderr, 6),
                    "R2": round(r**2, 4),
                    "p": f"{p:.2e}",
                    "Slope 95% CI": slope_ci,
                    "Intercept": round(intercept, 3),
                    "NW 95% CI": f"{ci_low_nw:.6f} – {ci_high_nw:.6f}",
                    "NW p": f"{nw_pval:.2e}",
                    "NW Slope Stderr": round(nw_se, 6),
                    "Maxlag": lags
                }
        
                heapq.heappush(heap, (r2, window_dict))
                if len(heap) > 4:
                    heapq.heappop(heap)  # remove smallest R²
        
            # Add top 4 windows for this channel to results
            results.extend([item[1] for item in heap])
        
        df_results = pd.DataFrame(results)
        return df_results
    
    def rolling_reg_ui(self, edited_df, results_df):
    
        if results_df.empty:
            st.warning("Not enough data for rolling regression.")
            return

        # Columns to show by default
        visible_cols = ["Channel", "Start", "End", "Slope", "R2"]
    
        # Build AgGrid options
        gb = GridOptionsBuilder.from_dataframe(results_df)
        for col in results_df.columns:
            # Hide columns not in visible_cols
            gb.configure_column(col, hide=(col not in visible_cols))
    
        # Enable selection
        gb.configure_selection("single", use_checkbox=True)
    
        # Enable the side panel to toggle columns
        gb.configure_side_bar(columns_panel=True)
    
        grid_options = gb.build()
    
        # Render AgGrid table
        grid_response = AgGrid(
            results_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=300,
            fit_columns_on_grid_load=True
        )
    
        selected = grid_response["selected_rows"]
        row = selected[0] if selected else None
    
        # Plotly chart if a row is selected
        plt.close('all')
        if row:
            sel_channel = row["Channel"]
            start, end = row["Start"], row["End"]
            slope, intercept = row["Slope"], row["Intercept"]
    
            fig = go.Figure()
            colors = px.colors.qualitative.T10
    
            # Plot all channels
            for idx, channel in enumerate(edited_df.columns[1:]):
                fig.add_trace(go.Scatter(
                    x=edited_df.iloc[:, 0],
                    y=edited_df[channel],
                    mode="lines",
                    name=channel,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    opacity=0.8
                ))
    
            # Highlight selected regression
            time = edited_df.iloc[:, 0]
            mask = (time >= start) & (time <= end)
            fig.add_trace(go.Scatter(
                x=time[mask],
                y=slope * time[mask] + intercept,
                mode="lines",
                name=f"Fit {row['Window']}",
                line=dict(color="black", width=3)
            ))
    
            # Shade the regression region
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="black", opacity=0.1, line_width=0
            )
    
            fig.update_layout(
                xaxis=dict(
                    title="Time (s)",
                    titlefont=dict(size=18, color="blue"),
                    tickfont=dict(size=14, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                yaxis=dict(
                    title="Signal",
                    titlefont=dict(size=18, color="blue"),
                    tickfont=dict(size=14, color="blue"),
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    mirror=True
                ),
                legend=dict(
                    title="Channels",
                    font=dict(size=14),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                ),
                width=900,
                height=400 + 100 * ((len(edited_df.columns)-1)//2),
                margin=dict(l=80, r=40, t=60, b=60)
            )
    
            st.plotly_chart(fig, use_container_width=True)

                         
    def download_separate(self, 
                      all_results, 
                      all_plots, 
                      rolling_reg_tabs=None, 
                      rolling_reg_plots=None):
        """
        Creates download buttons for each dataset.
        Each dataset gets:
          - one CSV (OLS results)
          - one PDF (OLS plots: raw + filtered)
          - (if available) one CSV (rolling regression results)
          - (if available) one PDF (rolling regression plots)
        """
    
        st.markdown("<p style='color: DarkRed; font-size: 20px;'>Download Results</p>", unsafe_allow_html=True)
    
        # --- OLS Results CSV ---
        for file_name, results_df in all_results.items():
            if results_df is None or results_df.empty:
                continue
    
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
    
            st.download_button(
                                label=f"Download OLS CSV for {file_name}",
                                data=csv_buffer.getvalue(),
                                file_name=f"{file_name}_OLS_results.csv",
                                mime="text/csv"
                                )
        # --- OLS Plots PDF ---
        file_keys = set(key.rsplit("_", 1)[0] for key in all_plots.keys())
        for file_name in file_keys:
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                raw_key = f"{file_name}_raw"
                if raw_key in all_plots:
                    pdf.savefig(all_plots[raw_key])
    
                filtered_key = f"{file_name}_filtered"
                if filtered_key in all_plots:
                    pdf.savefig(all_plots[filtered_key])
    
            pdf_buffer.seek(0)
    
            st.download_button(
                label=f"Download OLS Plots PDF for {file_name}",
                data=pdf_buffer,
                file_name=f"{file_name}_OLS_plots.pdf",
                mime="application/pdf"
            )

        # --- Rolling Regression Results CSV (optional) ---
        if rolling_reg_tabs:
            for file_name, roll_df in rolling_reg_tabs.items():
                if roll_df is None or roll_df.empty:
                    continue
    
                csv_buffer = io.StringIO()
                roll_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
    
                st.download_button(
                    label=f"Download Rolling Regression CSV for {file_name}",
                    data=csv_buffer.getvalue(),
                    file_name=f"{file_name}_rolling_regression.csv",
                    mime="text/csv"
                )
    
        # --- Rolling Regression Plots PDF (optional) ---
        if rolling_reg_plots:
            for file_name, figs in rolling_reg_plots.items():
                if figs is None:
                    continue
    
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:
                    # Always treat as list for flexibility
                    if not isinstance(figs, list):
                        figs = [figs]
                    for fig in figs:
                        pdf.savefig(fig)
                        plt.close(fig)
                        del fig
                        gc.collect()
    
                pdf_buffer.seek(0)
    
                st.download_button(
                    label=f"Download Rolling Regression Plots PDF for {file_name}",
                    data=pdf_buffer,
                    file_name=f"{file_name}_rolling_regression_plots.pdf",
                    mime="application/pdf"
                )
        
    def download_all_combined(self, 
                              all_results, 
                              all_plots,
                              rolling_reg_tabs=None, 
                              rolling_reg_plots=None):
        
        """
            Creates download buttons for:
             - a single CSV with all OLS results from all datasets
             - a single PDF containing all OLS plots
             - (if available) a single CSV with all rolling regression results
             - (if available) a single PDF containing all rolling regression plots
        """ 
    
        st.markdown("<p style='color: DarkRed; font-size: 20px;'>Download Combined Results</p>", unsafe_allow_html=True)
    
        # Download OLS results
        if all_results:
            combined_df = pd.concat(all_results.values(), ignore_index=True)
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
    
            st.download_button(
                                label="Download Combined OLS CSV",
                                data=csv_buffer.getvalue(),
                                file_name="combined_OLS_results.csv",
                                mime="text/csv"
                              )
    
        if all_plots:
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                # Sort keys to keep plots grouped per file
                for key in sorted(all_plots.keys()):
                    fig = all_plots[key]
                    if fig is not None:
                        pdf.savefig(fig)
            pdf_buffer.seek(0)
    
            st.download_button(
                label="Download Combined OLS Plots PDF",
                data=pdf_buffer,
                file_name="combined_OLS_plots.pdf",
                mime="application/pdf"
            )
    
        # Download Rolling Regression results (optional)
        if rolling_reg_tabs: # and any(rolling_reg_tabs.values()):
            combined_roll_df = pd.concat(rolling_reg_tabs.values(), keys=rolling_reg_tabs.keys())
            csv_buffer = io.StringIO()
            combined_roll_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
    
            st.download_button(
                                label="Download Combined Rolling Regression CSV",
                                data=csv_buffer.getvalue(),
                                file_name="combined_rolling_regression.csv",
                                mime="text/csv"
                              )
    
        if rolling_reg_plots: # and any(rolling_reg_plots.values()):
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                for key in sorted(rolling_reg_plots.keys()):
                    figs = rolling_reg_plots[key]
                    if figs is None:
                        continue
                    # Always treat as list for flexibility
                    if not isinstance(figs, list):
                        figs = [figs]
                    for fig in figs:
                        pdf.savefig(fig)
                        plt.close(fig)
                        del fig
                        gc.collect()
                        
            pdf_buffer.seek(0)
    
            st.download_button(
                                label="Download Combined Rolling Regression Plots PDF",
                                data=pdf_buffer,
                                file_name="combined_rolling_regression_plots.pdf",
                                mime="application/pdf"
                              )
        
    def download_plots_pdf(self, residuals_df=None, plots=None, key="plots", plot_type="residuals", max_lags=200):
        """
        Generate a PDF for downloading residuals and ACF plots:
          - If plots: dictionary of pre-made Matplotlib figures (e.g., rolling regression)
          - If residuals_df: generate plots based on plot_type
              * plot_type="residuals" → residuals vs time scatter
              * plot_type="acf"       → autocorrelation function (ACF)
        Returns a BytesIO buffer suitable for st.download_button.
        """
        pdf_buffer = io.BytesIO()
    
        with PdfPages(pdf_buffer) as pdf:
    
            # --- Case 1: Pre-made figures ---
            if plots:
                for fig in plots.values():
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)  # free memory
    
            # --- Case 2: Generate plots from residuals_df ---
            elif residuals_df is not None:
    
                # Flatten residuals efficiently
                residuals_df_flat = residuals_df.explode(["time", "residuals"])
                groups = residuals_df_flat.groupby(["File", "Channel"])
    
                for (file, ch), group in groups:
    
                    fig, ax = plt.subplots(figsize=(6, 4))
    
                    if plot_type == "residuals":
                        ax.scatter(group["time"], group["residuals"], s=15, alpha=0.6, color="steelblue")
                        ax.axhline(0, color="red", linestyle="--", linewidth=1)
                        ax.set_xlabel("Time (s)", fontsize=12, color="blue")
                        ax.set_ylabel("Residuals", fontsize=12, color="blue")
                    elif plot_type == "acf":
                        plot_acf(group["residuals"], lags=max_lags, ax=ax, alpha=0.05)
    
                    ax.set_title(f"{file} - {ch}" if plot_type=="residuals" else f"ACF: {file} - {ch}",
                                 fontsize=14, fontweight="bold", loc="center")
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                    ax.tick_params(axis='x', colors='blue', labelsize=10)
                    ax.tick_params(axis='y', colors='blue', labelsize=10)
    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)  # release memory
    
        pdf_buffer.seek(0)
        return pdf_buffer


    
class Utilities():
    """Contains static methods that can be accessed from all other classes."""
    
    def __init__(self):
        
        pass
    
    @staticmethod
    def add_to_state(state_variables, overwrite=False):
        """
        Instantiates session state variables from a dictionary.
        If overwrite=True, existing keys will be updated.
        """
        for key, value in state_variables.items():
            if overwrite or key not in st.session_state:
                st.session_state[key] = value


# Run 
if __name__ == '__main__':
    
    # Set up session_state variables
        
    if 'logo' not in st.session_state:
        st.session_state['logo'] = 'mote_logo.png'

    # Load image for favicon
    logo_img = Image.open(st.session_state['logo'])
        
    # Page config
    st.set_page_config(layout = "wide", 
                       page_title = 'Mote', 
                       page_icon = logo_img,
                       initial_sidebar_state="auto", 
                       menu_items = None)
    
    

    
    # Call Flow_Control class that makes all calls to other classes and methods
    obj1 = Flow_Control()
    all_calls = obj1.all_calls()
