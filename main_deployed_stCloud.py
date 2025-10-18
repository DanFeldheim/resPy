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
from numpy.lib.stride_tricks import sliding_window_view
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
from scipy.stats import norm
import math
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
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
import tempfile


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
         
        
    def all_calls(self):
        """This is the main logic workflow. All calls to other functions are here."""
        
        #---------------------------------------------------------------------------------------------
        # This prevents an error that occurs occasionally when the user attempts
        # to upload a file and the app hasn't loaded fully
        # For cloud use only
        # if "app_ready" not in st.session_state:
            
        # Mark app as not ready in state_variables_dict
        # Initialize session state variables and dicts
        state_variables_dict = {
                                # "app_ready": False,
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
                                "rolling_regression": False,
                                "tab1_files":[],
                                "tab2_files":[], 
                                "warning":False,
                                "uploaded_files_hash":None,
                                "snow":False,
                                "model_plots": False
                                }
            
        Utilities.add_to_state(state_variables_dict)
        
        
        # Instantiate dicts for storing data
        # Create dictionaries for data
        edited_raw_dfs_dict = {}
        final_x_y_valid_dict = {}
        all_regression_results = {}
        all_plots = {}
        rolling_reg_plots = {}
        
        # Master regression/residuals results dict
        all_subset_regs = {}
        all_subset_residuals = {}
       
        # Show loading message for first-time initialization
        st.write('')
        st.write('')
        st.write('')
     
        # Set up the header, login form, and check user credentials
        # Create Setup instance
        setup = Setup()
        
        # nav_bar = setup.navigation()
        
        # Render the header
        header = setup.header()
        
        if not st.session_state["snow"]:
            st.snow()
            st.session_state["snow"] = True
            
        
        # Create tabs
        tab1, tab2, tab3 = setup.tabs()

         
        #---------------------------------------------------------------------------------------------
        # Make calls to import, load, and analyze data
        
        with tab1:
            
            # st.write('')
            st.write('')
            
            # Call Load_Data class
            load_files = Load_Data()
    
            # Create an upload button to load file names to be analyzed
            file_name_list = load_files.upload()
             
            # st.write('')
            st.write('')
            st.divider()
            
            # If files were uploaded
            if file_name_list:
                
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>Enter Coral ID and Volume</p>",
                      unsafe_allow_html=True)
                    
                st.write('')
                
                # Call the Analysis class to plot data
                plot_data = Analysis()   
                
                # Loop through files and generate a dictionary of cleaned up x-y data dataframes
                # with filenames as keys and channel, time, and y data as columns
                for file in file_name_list:
                    
                    # Process the data; returns dataframe of raw data without NaNs
                    clean_filename, edited_df = self.preprocess_data(file, load_files, plot_data)
   
                    # Add the final raw edited_df to a dict 
                    edited_raw_dfs_dict[clean_filename] = edited_df
    
                
                # Create an edit table where user can enter coral ID and coral volume
                # Pass edited_raw_dfs_dict for filename and channel information
                volume_entry_df = plot_data.edit_table(edited_raw_dfs_dict)
                
                
                # Check that all rows have Coral ID and Volume filled
                missing_entries = volume_entry_df[volume_entry_df["Coral ID"].eq("") | volume_entry_df["Volume (mL)"].eq("")]
            
                # Continue when all volumes and coral IDs have been entered
                if missing_entries.empty:
                    
                    # Loop through  edited_raw_dfs_dict and do
                    # linear regression to get slope, R2, p-value, sum of squared residuals,
                    # slope standard error, and 95% CI
                    # Create an aggrid table for results
    
                    if len(edited_raw_dfs_dict) > 0:
                        
                        # key is filename, val is edited_df, and edited_df has time and channel cols
                        for key, val in edited_raw_dfs_dict.items():
                            
                            # Only keep valid channel columns
                            channels = [c for c in val.columns if c.startswith("Ch") and val[c].notna().any()]
                            
                            for ch in channels:
                                x = val['Time (s)'].values
                                start_time = float(np.min(x))
                                stop_time = float(np.max(x))
                                
                                # Get volume for the correct filename and channel from volume_entry_df 
                                volume = float(volume_entry_df.loc[(volume_entry_df['Filename'] == key)
                                                                   & (volume_entry_df['Channel'] == ch), 
                                                                   'Volume (mL)'].iloc[0])
                                
                                # Apply pressure correction of 1013/1013.25 = 0.99975
                                # Note this has too many sig figs compared to the channel data
                                # to be different from 1
                                # Also apply volume correction = vol/1000
                                
                                # Apply corrections directly to the DataFrame column so that
                                # edited_raw_dfs_dict is changed at all points downstream
                                val[ch] = val[ch] * 0.99975 * (volume / 1000)
                                
                                # Extract corrected y values for regression
                                y = val[ch].values
                    
                                # Fit regression
                                regress, residuals = plot_data.fit_regression(x, y)
                    
                                # Estimate maxlags as an indication of residual autocorrelation
                                maxlags = plot_data.estimate_maxlags(residuals)
                    
                                # Store results for all channels and filenames in one dict
                                all_regression_results.setdefault(key, {})[ch] = {
                                                                                # Unpack regression stats
                                                                                **regress,                 
                                                                                "Start Time": start_time,
                                                                                "Stop Time": stop_time,
                                                                                "Maxlags": maxlags
                                                                                }
                    
                    st.write("")
                    st.write("")
                    st.markdown(f"<p style='color: Blue; \
                          font-size: 24px; \
                          margin: 0;'>Preview Linear Regression Results</p>",
                          unsafe_allow_html=True)
                    
                    # Show results in aggrid table and return selected rows
                    selected_rows = plot_data.agtable(all_regression_results, volume_entry_df)
          
                    st.markdown(f"""
                                <p style='color: DarkRed; 
                                          font-size: 20px; 
                                          margin: 0;'>
                                          Select rows to view and adjust plots.<br>
                                          Then scroll to the bottom to download adjusted results table.
                                </p>
                                """, unsafe_allow_html=True)
                        
                    st.divider()
                    st.write('')
                
                    # Allow the user to select channels for further data processing and analysis
                    # Subset data based upon selected rows
                    subset_data = plot_data.subset_dataframe(selected_rows, edited_raw_dfs_dict)
                    
                    # Pass subset_data df to slider_bars() and plot functions
                    # Slider bars allow user to change the x-axis
                    if subset_data is not None:
                        for filename, df in subset_data.items():
                            
                            st.markdown(f"<p style='color: Black; \
                                          font-size: 24px; \
                                          margin: 0;'>{filename}</p>",
                                          unsafe_allow_html=True)
                        
                            # edited_df holds the data after user has changed the x-axis
                            # Here it is passed to a function that creates slider bars
                            # for each channel of each file to change x-axis
                            edited_df = plot_data.slider_bars(df, filename, tab1)
                            
                            # Plots and Returns new x and y values as a dict with
                            # channel as key and x_valid, y_valid as a list of 
                            # 1D arrays. e.g. ch1:[[x_valid], [y_valid]]
                            x_y_valid_dict, all_plots = plot_data.plot_channels(edited_df, 
                                                                                filename, 
                                                                                all_plots)
                            
                            # Store all x/y values for all filenames and channels
                            # after changing x-axis (or not)
                            final_x_y_valid_dict[filename] = x_y_valid_dict
                            
                            # Container for this file's channel regression results
                            channel_results = {}
                            
                            # Container for this file's channel residual results
                            channel_residuals = {}
                            
                            # Do the regression on the edited data
                            for channel, (x_valid, y_valid) in x_y_valid_dict.items():
                                
                                start_time = min(x_valid)
                                stop_time = max(x_valid)
                                
                                regress, residuals = plot_data.fit_regression(x_valid, y_valid)
                                maxlags = plot_data.estimate_maxlags(residuals)
                                
                                channel_results[channel] = {
                                                            **regress,
                                                            "Maxlags": maxlags,
                                                            "Start Time":start_time,   
                                                            "Stop Time":stop_time
                                                           }
                                
                                channel_residuals[channel] = residuals
                        
                            # Store all results in master dictionaries
                            all_subset_regs[filename] = channel_results
                            all_subset_residuals[filename] = channel_residuals
                            
                            st.markdown(
                                        f"<p style='color: DarkRed; font-size: 24px; margin: 0;'>Adjusted Linear Regression Results for {filename}</p>",
                                        unsafe_allow_html=True
                                        ) 
                            
                            # Generate an aggrid table that displays regression results for each 
                            # file under its plot
                            file_selected_rows = plot_data.agtable({filename: channel_results}, 
                                                                   volume_entry_df,
                                                                   use_checkboxes=False)
                            
                            st.divider()
                        
                        try:
                            # Displays a table of all regression results for selected channels
                            # and a button to download regression results.
                            download_reg = plot_data.download_regression(all_subset_regs, volume_entry_df)   
                        
                            # Download plots
                            download_plots = plot_data.download_respiration_plots(all_plots)
                            
                            st.divider()
                        
                        except:
                            pass
                        
              
            
        # Model Diagnostics calls
        #---------------------------------------------------------------------------------------------    
        with tab2:    
            
            st.write('')
  
            # If user selected files in the Linear Regression tab
            # then all_subset_regs will exist and we can continue
            if len(all_subset_regs) > 0:
                
                # Subset all_subset_regs
                # To keep
                keys_to_keep = ["slope", "R2", "slope stderr", "slope 95% CI", "slope pval"]
                
                # Make a new nested dictionary
                corrected_regression_dict = {}
                
                for filename, channels in all_subset_regs.items():
                    # Create an entry for this filename
                    corrected_regression_dict[filename] = {}
                
                    for ch, vals in channels.items():
                        # Create an entry for this channel
                        corrected_regression_dict[filename][ch] = {}
                
                        for k, v in vals.items():
                            if k in keys_to_keep:
                                corrected_regression_dict[filename][ch][k] = v  
                
                # Calculate newey-west regression parameters
                newey = plot_data.newey_west_analysis(final_x_y_valid_dict)
                
                # Merge newey and corrected_regression_dict
                merge_nw = plot_data.merge_nested_dicts(corrected_regression_dict, newey)
                
                # Perform an ESS correction
                ess_correction = plot_data.ESS_correction_from_residuals(all_subset_residuals)
                
                # Merge ess_correction and merge_nw
                final_corrected_results_dict = plot_data.merge_nested_dicts(ess_correction, merge_nw)
                
                # Reorder and round
                final_corrected_results_dict = plot_data.format_regression_results(final_corrected_results_dict)
                
                # Convert to df for display in a data_editor table
                rows = []
                for fname, channels in final_corrected_results_dict.items():
                    for ch, metrics in channels.items():
                        row = {"filename": fname, "channel": ch}
                        row.update(metrics)
                        rows.append(row)
                
                final_corrected_df = pd.DataFrame(rows)
                
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>Uncorrected and Newey-West Corrected Regression Results</p>",
                      unsafe_allow_html=True)
                
                st.write('')
                
                # Display
                st.data_editor(final_corrected_df,
                                use_container_width=True,
                                num_rows="static",
                                hide_index=True
                              )
                
                # Download
                csv = final_corrected_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                                    label="Download Corrected Regression Results",
                                    data=csv,
                                    file_name="corrected_regression_results.csv",
                                    mime="text/csv",
                                  )
                
                st.write('')
                st.divider()

                col1, col2 = st.columns([1,1])

                with col1:
                    # View all acf, residuals, and normal distribution plots
                    st.markdown(
                                """
                                <div style='text-align: left; color: blue; font-size:24px;'>
                                    View ACF, Residuals, and Normal Distribution Plots
                                </div>
                                """,
                                unsafe_allow_html=True
                                )
                with col2:
                    self.about_model_diagnostics()
                st.write("") 
                
                model_plots = st.button("Model Diagnostics Plots")
 
                if model_plots:
                    
                    st.divider()
                    
                    # Call residuals, acf, and normal distribution plots
                    figs = plot_data.generate_plots(all_subset_residuals, final_x_y_valid_dict)
                    
                    # Prevent memory leak
                    gc.collect()
                                
            else:
                
                # Must select data to analyze in linear regression tab first
                col1, col2, col3 = st.columns([0.5,6,1])
                
                with col2:
                
                    st.markdown(f"<p style='color: Red; \
                          font-size: 24px; \
                          margin: 0;'>Please go to the Linear Regression tab, upload data, and select channels to analyze.</p>",
                          unsafe_allow_html=True)
                        
            with tab3:    
                
                st.write('')
      
                # If user selected files in the Linear Regression tab
                # then all_subset_regs will exist and we can continue
                if len(all_subset_regs) > 0:
                    
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

                    st.markdown(f"<p style='color: black; \
                                  font-size: 14px; \
                                  margin: 0;'>Select a channel to view plot and fit line.</p>",
                                  unsafe_allow_html=True)
                        
                     
                    # Computes regression fits for small windows in the data
                    rolling_reg = plot_data.compute_rolling_regression(final_x_y_valid_dict, window_size)
                    
                    # Displays rolling regression table and plot
                    rolling_reg_plots = plot_data.rolling_reg_ui(final_x_y_valid_dict, rolling_reg)
          
                
                else:
                    
                    col1, col2, col3 = st.columns([0.5,6,1])
                    
                    with col2:
                    
                        st.markdown(f"<p style='color: Red; \
                              font-size: 24px; \
                              margin: 0;'>Please go to the Linear Regression tab, upload data, and select channels to analyze.</p>",
                              unsafe_allow_html=True)
                            
                            
                
    #------------------------------------------------------------------------------------------------------------
    # Some "helper" functions to the flow control class
    def preprocess_data(self, file, load_files, plot_data):
        """Helper function to the flow control class to clean up the data before calculations."""

        # Get file name
        file_name = file.name
        
        # Strip out .csv from file name
        clean_filename, _ = os.path.splitext(file_name)
        
        # Import into dataframe
        import_data = load_files.import_df(file)
        
        # Grab time and oxygen concentration columns
        oxy_df = load_files.oxy_data(import_data)
        
        # Convert everything to numeric (non-numeric → NaN)
        oxy_numeric = oxy_df.apply(pd.to_numeric, errors='coerce')
        
        # Drop columns where all values are NaN
        oxy_column_cleaned = oxy_numeric.dropna(axis=1, how='all')
        
        # Drop rows where any column is NaN
        edited_df = oxy_column_cleaned.dropna(axis=0, how='any')
       

        return clean_filename, edited_df
    
    def about_model_diagnostics(self):
        """A helper function to the flow control class that explains the various model diagnostics."""
        
        with st.expander('🪸 About Model Diagnostics'):
            
            st.markdown(
                        """
                        ### 🎛️ Noise Analysis
                        - **Residuals Plots** → Residuals should be scattered randomly around the 0 line.
                        A non-random pattern in the residuals indicates autocorrlation or heteroskedasticity.
                        This violates one of the assumptions of linear regression and indicates sluggishness
                        in the experiment (slow mixing, sensor response). A consequence of residual correlation
                        is that standard errors, confidence intervals, and p-values are underestimated. 
                        These can be corrected with a Newey-West correction (see Signal Processing).
                        - **Autocorrelation Function Plots** → Shows the strength and direction of the 
                        correlation from -1 to 1 with 95% CI. The x axis represents the correlation of
                        a data point with other points n "lags" or time periods preceding that point. Any
                        bar above the 95% CI regions is statistically autocorrelated.
                        - **Normality Tests** → The histogram of residuals should be normally distributed.
                        This is another fundamental assumption for linear regression. 
                        
                        ---
                        
                        ### 📈 Signal Processing
                        - **Correlation Correction** → When residuals are correlated, the usual error bars 
                        and p-values become too optimistic. The Newey-West method corrects this by widening 
                        the error bars so they better reflect reality, and it should be used when reporting 
                        slopes, confidence intervals, and p-values. The Effective Sample Size (ESS) method 
                        takes a different approach: it reduces the number of “independent” data points to 
                        account for correlation, then recalculates the error bars and p-values. ESS is best 
                        treated as a diagnostic — a small ESS suggests sluggish or delayed responses in 
                        the experiment — while Newey-West provides the corrected statistics you should 
                        use for inference calculations such as experimental differences.
                        """,
                        unsafe_allow_html=True
                        )  
        
        #---------------------------------------------------------------------------------------------
        # Setup class
        
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
        col1, col2 = st.columns([1,6])
        
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
        
    def tabs(self):
        """Styles and creates the tab bar."""
        
        # Style the tabs
        st.markdown("""
                    <style>
                    /* target the tab-list inside the Streamlit tab wrapper */
                    .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]{
                        font-weight: 700 !important;
                        color: #1E90FF !important;
                        border-bottom: 3px solid #1E90FF !important;
                        background-color: #f0f2f6 !important;
                        border-radius: 8px 8px 0 0 !important;
                    }
                
                    .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="false"]{
                        color: gray !important;
                        background: transparent !important;
                    }
                
                    .stTabs [data-baseweb="tab-list"] button[role="tab"]{
                        padding: 8px 14px !important;
                        transition: transform .12s ease, color .12s ease !important;
                    }
                    .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover{
                        transform: scale(1.03) !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
        st.markdown("""
                    <style>
                    /* Tab text (normal and active) */
                    .stTabs [data-baseweb="tab-list"] button[role="tab"] > div {
                        font-size: 18px !important;  /* increase font size */
                        font-weight: 600 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
    
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Linear Regression", "Model Diagnostics", "Rolling Regression"])
        
        return tab1, tab2, tab3

        
    def navigation(self):
        """Creates a sidebar only for the purposes of checking memory leak. Otherwise not called."""
        
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
    
    def upload(self): #, key):
        
        # File uploader that allows multiple files
        uploaded_files = st.file_uploader(
                                          "Choose files", 
                                          type=["csv", "txt"],  
                                          accept_multiple_files=True, 
                                          # key = key
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
                df = pd.read_csv(file, sep="\t", encoding='latin1', skiprows=19)
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
        oxy_df.columns = ['Time (s)', 'Ch1', 'Ch2', 'Ch3', 'Ch4']

        return oxy_df
    
    
class Analysis():
    
    """Performs all calculations-linear regression, noise, 
    and model analysis-and generates plots, tables, and download buttons."""
    
    def __init__(self):
        
        pass
    
    def raw_data_table(self, oxy_df, file_name):
        """Cleans the dataframe and displays the data in st.data_editor."""
        
        st.write('')
        st.write('')
        
        col1, col2, col3 = st.columns([1,1,1])
        
        with col2: 
            
            # Write the filename
            st.markdown(f"<p style='color: DarkRed; \
                  font-size: 24px; \
                  margin: 0;'>{file_name}</p>",
                  unsafe_allow_html=True)
        
        st.write('')
        
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            
            st.markdown(f"<p style='color: DarkRed; \
                  font-size: 24px; \
                  margin: 0;'>Raw Data</p>",
                  unsafe_allow_html=True)
        
            # Create a data table  
            col_names = oxy_df.columns
            
            # Create column_config for all columns as "small"
            column_config = {
                            col: st.column_config.Column(col, width=100)
                            for col in oxy_df.columns
                            }
                
            raw_data_df = st.data_editor(oxy_df, 
                                       use_container_width=False,
                                       num_rows="static",
                                       hide_index=True,
                                       height=250,
                                       column_config=column_config,
                                       key=f"raw{file_name}"
                                       )
        
        # Drop rows with NaN 
        naScrubbed_raw_data_df = raw_data_df.dropna(subset=['Time (s)', 'Ch1', 'Ch2', 'Ch3', 'Ch4'])

        del raw_data_df
        gc.collect()
        
        return naScrubbed_raw_data_df
    
    def edit_table(self, edited_raw_dfs_dict):
        """
        Step 1: Create a simple editable table for entering Coral ID and Volume (mL).
        Columns: Filename, Channel, Coral ID, Volume (mL)
        Uses st.data_editor
        """
        
        # Flatten the nested results into a list of rows
        rows = []
        for filename, channels in edited_raw_dfs_dict.items():
            for ch in channels.keys():
                if ch.lower().startswith("ch"):
                    row = {
                            "Filename": filename,
                            "Channel": ch,
                            "Coral ID": "",
                            "Volume (mL)": "",
                            # Need to add a dummy col to make the drag contents feature
                            # work in the volume column
                            "_dummy": "",
                          }
                    
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        col1, col2, col3 = st.columns([1,4,1])
        
        with col2:
            
            # Add to edit table
            volume_table_df = st.data_editor(df,
                                            use_container_width=False,
                                            num_rows="static",
                                            hide_index=True,
                                            column_config=
                                            {
                                            "Filename": st.column_config.TextColumn(
                                            "Filename",
                                            width="medium"
                                            ),
                                            "Channel": st.column_config.TextColumn(
                                            "Channel",
                                            width='medium' 
                                            ),
                                            "Coral ID": st.column_config.TextColumn(
                                            "Coral ID",
                                            width='medium'
                                            ),
                                            "Volume (mL)": st.column_config.TextColumn(
                                            "Volume (mL)",
                                            width='medium' 
                                            ),
                                            "_dummy": st.column_config.TextColumn(
                                            "", 
                                            width="small")
                                            }
                                            )

        # Return the table and drop the dummy col
        return volume_table_df.drop(columns=["_dummy"], errors="ignore")
    
    def agtable(self, all_regression_results, 
                data_editor_df=None,
                base_row_height: int = 30, 
                max_rows: int = 20, 
                use_checkboxes: bool = True):
        """
        Render an AgGrid table of regression results.
    
        Parameters
        ----------
        all_regression_results : dict
            Nested dict {filename: {channel: regression_results}}.
        table height parameters
        use_checkboxes : bool, optional
            Whether to show checkboxes for row selection (default = True).
        """

        # Flatten the nested dictionaries into a list of rows
        rows = []
        # Get values from all_regression_results
        for filename, channels in all_regression_results.items():
            for ch, results in channels.items():
                
                # Look up Coral ID and Volume from data_editor_df if provided
                if data_editor_df is not None:
                    match = data_editor_df[
                                            (data_editor_df["Filename"] == filename) &
                                            (data_editor_df["Channel"] == ch)
                                          ]
                    coral_id = match["Coral ID"].values[0] if not match.empty else ""
                    vol = match["Volume (mL)"].values[0] if not match.empty else ""
                    
                else:
                    coral_id = "No ID Entered"
                    vol = "No Coral Volume Entered"
                    
                maxlag = results.get("Maxlags")
                # Get regression results
                slope = results.get("slope")
                slope_ci = results.get("slope 95% CI")
                R2 = results.get("R2")
                SSR = results.get("squared residuals")
                pval = results.get("slope pval")
                stderr = results.get("slope stderr")
    
                # Round numeric values
                slope = round(slope, 4) if isinstance(slope, (float,int)) else np.nan
                # slope_ci = round(slope_ci, 4) if isinstance(slope_ci, (float,int)) else np.nan
                R2 = round(R2, 3) if isinstance(R2, (float,int)) else np.nan
                SSR = round(SSR, 2) if isinstance(SSR, (float,int)) else np.nan
                maxlag = maxlag if isinstance(maxlag, (float,int)) else np.nan
    
                # Format p-value and stderr for display
                pval_str = f"{pval:.2e}" if isinstance(pval,(float,int)) else ""
                stderr_str = f"{stderr:.2e}" if isinstance(stderr,(float,int)) else ""
                
                # Build a dictionary
                row = {
                        "Filename": filename,
                        "Channel": ch,
                        "Coral ID": coral_id,
                        "Volume (mL)": vol,
                        "Start Time": round(results.get("Start Time", np.nan),2) if isinstance(results.get("Start Time", np.nan),(float,int)) else np.nan,
                        "Stop Time": round(results.get("Stop Time", np.nan),2) if isinstance(results.get("Stop Time", np.nan),(float,int)) else np.nan,
                        "slope": slope,
                        "slope 95% CI": slope_ci,
                        "R2": R2,
                        "slope pval": pval_str,
                        "p-value_numeric": pval if isinstance(pval,(float,int)) else np.nan,
                        "slope stderr": stderr_str,
                        "SSR": SSR,
                        "MaxLag": maxlag
                      }
                
                # Append to rows list
                rows.append(row)
                
        # Build a dataframe from the results
        df = pd.DataFrame(rows)
        
        # Dynamic height calculation with +1 for the header
        num_rows = len(df)
        visible_rows = min(num_rows, max_rows)
        height = (visible_rows + 1) * base_row_height
    
        # Build AgGrid table
        gb = GridOptionsBuilder.from_dataframe(df)
        
        # Only enable checkbox selection if requested
        if use_checkboxes:
            gb.configure_selection("multiple", 
                                   use_checkbox=True,
                                   header_checkbox=True
                                   )
            
        else:
            gb.configure_selection("single", 
                                   use_checkbox=False,
                                   header_checkbox=True
                                   )
    
        # JS code for conditional coloring
        cell_style_jscode = JsCode("""
                                    function(params) {
                                        if(params.data['p-value_numeric'] > 0.05) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        if(params.value != null && params.colDef.field == 'R2' && params.value < 0.95) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        if(params.value != null && params.colDef.field == 'SSR' && params.value > 1000) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        if(params.value != null && params.colDef.field == 'MaxLag' && params.value > 20) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        return null;
                                    };
                                    """)
    
        # Apply JS styling to each column that needs it
        highlight_columns = ["slope pval", "R2", "SSR", "MaxLag"]
        
        for col in highlight_columns:
            gb.configure_column(col, cellStyle=cell_style_jscode)
    
        # Hide the helper numeric column
        gb.configure_column("p-value_numeric", hide=True)
    
        grid_options = gb.build()
        
        grid_response = AgGrid(
                                df,
                                gridOptions=grid_options,
                                allow_unsafe_jscode=True,  
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                fit_columns_on_grid_load=True,
                                height=height
                              )
        
        # Show MaxLag warning in a popup window if needed
        # Controlled by session_state['warning'] so that it only runs once
        # per upload
        # Try/Except catches an error if Ok isn't clicked before moving to another tab
        try:
            if not st.session_state["warning"] and (df["MaxLag"] > 20).any():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                # Use a placeholder to control exact layout
                placeholder = col2.empty()  
            
                with placeholder.container():
                    st.error(
                            "Warning! Maxlags are too high for the channels highlighted.\n"
                            "This indicates high autocorrelation in the data that yields low estimates\n"
                            "for the standard error.\n"
                            "This can be rectified with a Newey-West correction in the Model Diagnostics tab."
                            )
                    
                    # Button click sets warning to True
                    if st.button("Ok", key="maxlag_warning"):
                        st.session_state["warning"] = True
                        placeholder.empty()  

        except st.errors.StreamlitDuplicateElementKey:
            pass
        
        # Return only the rows selected by the user
        selected_rows = grid_response["selected_rows"]
        
        return selected_rows
    
    def subset_dataframe(self, selected_rows, edited_raw_dfs_dict):
        """Takes in the filenames and channels selected from the aggrid table
        and pulls the x-y data from edited_raw_dfs_dict."""
        
        # Subset the original dataframe to get the rows selected
        # for further analysis
        # Dict to hold data
        subset_data = {}
    
        if selected_rows is not None:
            # Loop through selected_rows df
            for _, row in selected_rows.iterrows():
                # Get filename and channel
                filename = row["Filename"]
                channel = row["Channel"]

                # Get the full dataframe for this file
                df = edited_raw_dfs_dict[filename]
                
                # Get the channel for this filename
                if df is not None and channel in df.columns:
                    # Subset time + this channel
                    subset_df = df[["Time (s)", channel]].dropna()
        
                    # If filename not seen yet, create a key with the filename in subset_data,
                    # otherwise merge the channel with the existing filename in subset_data.
                    # Could also have stuck identical filename keys and then grouped.
                    if filename not in subset_data:
                        subset_data[filename] = subset_df.rename(columns={channel: channel})
                        
                    else:
                        # Merge with existing dataframe for this file
                        subset_data[filename] = pd.merge(
                                                        subset_data[filename],
                                                        subset_df.rename(columns={channel: channel}),
                                                        on="Time (s)",
                                                        how="outer"
                                                        )
        
            # Sort and clean up each file’s dataframe
            for fname in subset_data:
                subset_data[fname] = subset_data[fname].sort_values("Time (s)").reset_index(drop=True)
        
            
        return subset_data if subset_data else None                          
    
    def slider_bars(self, raw_data_dropped_channels, file_name, tab):
        
        """
        Adds independent sliders for each channel in the dataframe
        and returns a new dataframe where each channel is masked to
        its own time window.
        
        Parameters
        ----------
        raw_data_dropped_channels : pd.DataFrame of the raw data selected by the user
            from the aggrid table of all files. Comes from subset_data
            Must contain a 'time (s)' column plus one or more channel columns.
        file_name : str
            Unique identifier for Streamlit widget keys.
        tab: Whether the call came from tab1, tab2, or tab3
                needed for the slider key
        
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
        time = edited_df['Time (s)']
        
        # Get existing channel cols since some may be deleted
        channel_cols = [c for c in edited_df.columns if c != 'Time (s)']
    
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
                                            key=f"time_slider_{file_name}_{col}_{tab}"
                                            )
                    # Create a mask that includes the times the user wants to keep
                    mask = (time >= t_min) & (time <= t_max)
                    # Set all times outside the mask to nan
                    edited_df.loc[~mask, col] = float('nan')
     
        return edited_df
     
    def plot_channels(self, edited_df, file_name, all_plots):
        """
        Plot all channels in edited_df.
        Automatically handles NaNs created if the slider was used
        to change the x-axis.
    
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
        time = edited_df["Time (s)"].values
       
        # Create a dict with ch as key and x_valid and y_valid as values
        x_y_valid_dict = {}
        
        # Make sure any existing plots are closed
        plt.close("all")
        
        # Create plot instance size
        figsize = (4, 3)
    
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
            for ch in edited_df.drop(columns=["Time (s)"]).columns:
                
                # Grab y values
                y = edited_df[ch].values
    
                # Mask NaNs (deleted points via sliders)
                mask = ~np.isnan(y)
                # These are the x and y values the user wanted to keep
                # according to the position of the slider bar
                x_valid = time[mask]
                y_valid = y[mask]
                
                x_y_valid_dict[ch] = [x_valid, y_valid] 
                
                # Get low and high times to do a time range column in results
                low_time = x_valid.min()
                high_time = x_valid.max()
    
                # Plot only valid points
                ax.plot(x_valid, y_valid, label=ch)
            
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
        
        return x_y_valid_dict, all_plots
        
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
            slope_ci = f"'{ci_low:.6f} - {ci_high:.6f}"
        
        else:
            slope_ci = "NA"
            
        # Calculate R2
        R2 = r*r
        
        # Calculate sum of squared residuals
        # Predict y values from regression
        y_pred = intercept + slope * np.array(x)
        # Calculate residuals
        residuals = np.array(y) - y_pred
        # Calculate sum of squared residuals
        ssr = np.sum(residuals**2)
        
        regress_results = {
                            "slope":slope,
                            "intercept":intercept,
                            "R2":R2,
                            "slope pval":p,
                            "slope stderr":stderr,
                            "slope 95% CI":slope_ci,
                            "squared residuals":ssr
                          }
        
        return regress_results, residuals

    def estimate_maxlags(self, residuals, max_lag=200):
        """
        Estimate maxlags for a single residuals array.
        """
        
        # Call acf function from statsmodel
        acf_vals = acf(residuals, nlags=max_lag, fft=True)
        N = len(residuals)
        threshold = 2 / np.sqrt(N)
    
        best_lag = max_lag
        for lag in range(1, len(acf_vals)):
            if abs(acf_vals[lag]) < threshold:
                best_lag = lag
                break
    
        return best_lag
    
    def newey_west_analysis(self, final_x_y_valid_dict, alpha=0.05, max_lag=200):
        """
        Run OLS regressions on nested x-y data and apply Newey-West correction.
        
        Parameters
        ----------
        self : object
            Class instance that has estimate_maxlags(residuals, max_lag).
        final_x_y_valid_dict : dict
            Nested dictionary of the form:
            {
                "file1": {"ch1": (x_array, y_array), "ch2": (x_array, y_array)},
                "file2": {"ch1": (x_array, y_array)}
            }
        alpha : float
            Significance level for CI (default 0.05 → 95% CI).
        max_lag : int
            Upper bound for lag search (default 200).
        
        Returns
        -------
        results_dict : dict
            Dictionary with slope, intercept, HAC-corrected SE, p-value, CI, and lags.
        """
        
        # Dcit to store results
        results_dict = {}
        # Calculate z score with scikit learn (2-tailed, 95% CI)
        z = norm.ppf(1 - alpha/2)
        
        # Loop through x-y data for each flename and channel
        for fname, channels in final_x_y_valid_dict.items():
            # results for each filename
            results_dict[fname] = {}
            
            # Get x-y data for each channel
            for ch, (x, y) in channels.items():
                x = np.asarray(x)
                y = np.asarray(y)
                
                # Build regression matrix
                X = sm.add_constant(x)  
                ols_model = sm.OLS(y, X).fit()
                
                # Residuals for lag estimation
                residuals = ols_model.resid
                # Estimate the max lag for NW correction
                lags = self.estimate_maxlags(residuals, max_lag=max_lag)
                
                # HAC regression
                nw_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
                
                # intercept = nw_model.params[0]
                slope = nw_model.params[1]
                se_slope = nw_model.bse[1]
                pval_slope = "{:.3e}".format(nw_model.pvalues[1])
                r2 = nw_model.rsquared
                
                ci_low = slope - z * se_slope
                ci_high = slope + z * se_slope
                
                nw_slope_ci = f"'{ci_low:.6f} - {ci_high:.6f}"
                
                results_dict[fname][ch] = {
                    # "intercept": intercept,
                    "NW slope": slope,
                    "NW slope stderr": se_slope,
                    "NW slope pval": pval_slope,
                    "NW slope 95% CI": nw_slope_ci,
                    "Lags": lags
                    }
        
        return results_dict
    
    def ESS_correction_from_residuals(self, residuals_dict, max_lag=200):
        """
        Compute effective sample size (ESS) from precomputed residuals.
        
        Parameters
        ----------
        self : object
            Class instance with method estimate_maxlags(residuals, max_lag)
        residuals_dict : dict
            Nested dictionary {filename: {channel: residuals_array}}
        max_lag : int
            Maximum lag to consider when estimating ESS
        
        Returns
        -------
        ess_dict : dict
            Nested dictionary with same structure containing ESS value.
        """
        
        # Dict to store results
        ess_dict = {}
        
        # Loop through filenames and channels and get residuals
        for fname, channels in residuals_dict.items():
            ess_dict[fname] = {}
            
            # Loop through each channel in channels
            for ch, residuals in channels.items():
                
                # Get residuals as numpy array
                residuals = np.asarray(residuals)
                N = len(residuals)
                
                # Determine max lag
                lags = self.estimate_maxlags(residuals, max_lag=max_lag)
                
                # Compute autocorrelations up to lag
                acf_vals = acf(residuals, nlags=lags, fft=True)[1:]  # skip lag 0
                
                # Compute effective sample size
                ess = N / (1 + 2 * np.sum((N - np.arange(1, len(acf_vals)+1)) / N * acf_vals))
                
                ess_dict[fname][ch] = {"ESS": ess}
        
        return ess_dict
        
    def merge_nested_dicts(self, dict1, dict2):
        """
        Merge two nested dictionaries with the same structure.
        dict2 values overwrite dict1 values if keys match.
        """
        merged = {}
        for fname in dict1:
            merged[fname] = {}
            for ch in dict1[fname]:
                merged[fname][ch] = {}
                # Combine keys from both dicts
                keys = set(dict1[fname][ch].keys()) | set(dict2.get(fname, {}).get(ch, {}).keys())
                for key in keys:
                    if key in dict2.get(fname, {}).get(ch, {}):
                        merged[fname][ch][key] = dict2[fname][ch][key]
                    else:
                        merged[fname][ch][key] = dict1[fname][ch][key]
        return merged
    
    def format_regression_results(self,results_dict):
        """
        Reorder keys and format numbers for display.
        
        Parameters
        ----------
        results_dict : dict
            Nested dictionary of the form {filename: {channel: {metrics}}}
        
        Returns
        -------
        formatted_dict : dict
            Nested dictionary with reordered keys, rounded numbers, and scientific p-values.
        """
        
        key_order = [
            "slope",
            "R2",
            "slope stderr",
            "slope pval",
            "slope 95% CI",
            "NW slope",
            "NW slope stderr",
            "NW slope pval",
            "NW slope 95% CI",
            "Lags",
            "ESS"
        ]
        
        formatted_dict = {}
        
        for fname, channels in results_dict.items():
            formatted_dict[fname] = {}
            for ch, metrics in channels.items():
                new_metrics = {}
                
                # Round / format values
                for key in metrics:
                    val = metrics[key]
                    if key in ["slope", "NW slope"]:
                        val = round(val, 4)
                    elif key == "R2":
                        val = round(val, 3)
                    elif key in ["slope stderr", "NW slope stderr"]:
                        # val = round(val, 4)
                        val = "{:.3e}".format(val)
                    elif key in ["ESS"]:
                        val = round(val, 1)
                    # pvals seem to be strings, convert to float first
                    elif key in ["slope pval", "NW slope pval"]:
                        if isinstance(val, (int, float)):
                            val = "{:.3e}".format(val)
                    new_metrics[key] = val
                
                # Reorder keys
                ordered_metrics = {k: new_metrics[k] for k in key_order if k in new_metrics}
                formatted_dict[fname][ch] = ordered_metrics
                
        return formatted_dict
    
    def plot_residuals(self, residuals, ax):
        """Residuals vs index around 0"""
        ax.scatter(np.arange(len(residuals)), residuals, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Index")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals around 0")
    
    def plot_residual_acf(self, residuals, ax, estimate_maxlags_func, max_lag=200):
        """
        ACF plot using the provided estimate_maxlags function.
        Pass in your own self.estimate_maxlags method.
        """
        best_lag = self.estimate_maxlags(residuals, max_lag=max_lag)
        plot_acf(residuals, lags=best_lag, ax=ax, zero=False)
        ax.set_title(f"ACF of Residuals (maxlag={best_lag})")
    
    def plot_residual_distribution(self, residuals, ax):
        """Residual histogram with fitted normal curve"""
        ax.hist(residuals, bins=20, density=True, alpha=0.6, color="gray")
        mu, std = norm.fit(residuals)
        x = np.linspace(min(residuals), max(residuals), 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, "r--", linewidth=2)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
    
    def generate_plots(self, 
                       all_subset_residuals, 
                       estimate_maxlags_func, 
                       max_lag=200, 
                       pdf_filename="model_diagnostics.pdf"):
        """
        Display 3-column plots for each file-channel in Streamlit and save all plots into a single PDF.
        Figures are closed immediately after display to prevent memory leaks.
        Figure titles ensure filename and channel appear in PDF.
        """
        # Create a PDF to save all figures
        with PdfPages(pdf_filename) as pdf:
            for filename, channels in all_subset_residuals.items():
                for channel, residuals in channels.items():
                    # Streamlit header
                    # st.markdown(f"### {filename} — {channel}")
    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
                    # Add figure-wide title so it appears in PDF
                    fig.suptitle(f"{filename} — {channel}", fontsize=14)
    
                    # Plot the 3 subplots
                    self.plot_residuals(residuals, axes[0])
                    self.plot_residual_acf(residuals, axes[1], estimate_maxlags_func, max_lag=max_lag)
                    self.plot_residual_distribution(residuals, axes[2])
    
                    # Adjust layout to leave space for suptitle
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
                    # Display in Streamlit
                    st.pyplot(fig)
    
                    # Save figure to PDF
                    pdf.savefig(fig)
    
                    # Close figure to free memory
                    plt.close(fig)
                    del fig, axes
                    gc.collect()
                    
        # Read PDF bytes for download
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
    
        st.download_button(
                            label="Download and Close Plots",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf"
                          )
        
        # To prevent memory leak
        # Remove the temporary PDF file after it's been used
        try:
            os.remove(pdf_filename)
            
        except FileNotFoundError:
            pass
        
        del pdf_bytes
        gc.collect()
    
    @st.cache_data
    def compute_rolling_regression(_self,
                                   final_x_y_valid_dict,
                                   window_size,
                                   step_perc=5):
        """
        Vectorized rolling regression using numpy.
        Accepts nested dictionary:
            {filename: {channel: [x_array, y_array]}}
        Returns: DataFrame of rolling regression results.
        """
    
        results = []
    
        # Progress bar
        progress = st.progress(0)
        total_files = len(final_x_y_valid_dict)
        file_idx = 0
    
        for filename, channels in final_x_y_valid_dict.items():
            for channel, xy in channels.items():
                x, y = np.array(xy[0]), np.array(xy[1])
    
                step = max(1, int(window_size * step_perc * 0.1))
                
                if len(x) < window_size:
                    continue
    
                # Create rolling windows (shape: num_windows x window_size)
                x_windows = sliding_window_view(x, window_shape=window_size)[::step]
                y_windows = sliding_window_view(y, window_shape=window_size)[::step]
    
                num_windows = x_windows.shape[0]
                heap = []
    
                # Vectorized OLS for each window
                for i in range(num_windows):
                    xi = x_windows[i]
                    yi = y_windows[i]
    
                    mask = ~np.isnan(yi)
                    xi, yi = xi[mask], yi[mask]
                    if len(xi) < 2:
                        continue
    
                    # OLS fit (single fit)
                    X = sm.add_constant(xi)
                    ols_model = sm.OLS(yi, X).fit()
                    slope = ols_model.params[1]
                    intercept = ols_model.params[0]
                    r2 = ols_model.rsquared
                    stderr = ols_model.bse[1]
                    residuals = ols_model.resid
                    ssr = np.sum(residuals**2)
                    ci_low = slope - 1.96 * stderr
                    ci_high = slope + 1.96 * stderr
                    slope_ci = f"{ci_low:.6f} - {ci_high:.6f}"
    
                    # Newey-West robust SE
                    lags = _self.estimate_maxlags(residuals, max_lag=200)
                    nw_model = ols_model.get_robustcov_results(cov_type="HAC", maxlags=lags)
                    nw_slope = nw_model.params[1]
                    nw_se = nw_model.bse[1]
                    nw_pval = nw_model.pvalues[1]
                    ci_low_nw = nw_slope - 1.96 * nw_se
                    ci_high_nw = nw_slope + 1.96 * nw_se
    
                    window_dict = {
                        "Filename": filename,
                        "Channel": channel,
                        "Window": f"{xi[0]:.1f} - {xi[-1]:.1f}",
                        "Start": xi[0],
                        "End": xi[-1],
                        "Slope": round(slope, 4),
                        "intercept": round(intercept, 4),
                        "Slope Stderr": round(stderr, 6),
                        "R2": round(r2, 4),
                        "Slope pval": f"{nw_model.pvalues[1]:.2e}",
                        "Slope 95% CI": slope_ci,
                        "SSR": round(ssr, 3),
                        "NW Slope 95% CI": f"{ci_low_nw:.6f} - {ci_high_nw:.6f}",
                        "NW pval": f"{nw_pval:.2e}",
                        "NW Slope Stderr": round(nw_se, 6),
                        "Lag": lags
                    }
    
                    heapq.heappush(heap, (r2, window_dict))
                    if len(heap) > 4:
                        heapq.heappop(heap)
    
                # Sort top windows by R²
                heap.sort(reverse=True)
                results.extend([item[1] for item in heap])
    
            file_idx += 1
            progress.progress(file_idx / total_files)
    
        return pd.DataFrame(results)
    
    def rolling_reg_ui(self, 
                   final_x_y_valid_dict, 
                   results_df,
                   base_row_height: int = 30, 
                   max_rows: int = 20):
        
        if results_df.empty:
            st.warning("Not enough data for rolling regression.")
            return
    
        # If you don't want to show all the columns in the df, you can
        # put some in a side bar.
        # Columns to show by default
        # visible_cols = ["Filename", "Channel", "Start", "End", "Slope", "R2"]
    
        # Build AgGrid options
        gb = GridOptionsBuilder.from_dataframe(results_df)
        # for col in results_df.columns:
        #     gb.configure_column(col, hide=(col not in visible_cols))
    
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_side_bar(columns_panel=True)
        grid_options = gb.build()
        
        # Dynamic height calculation with +1 for the header
        num_rows = len(results_df)
        visible_rows = min(num_rows, max_rows)
        height = (visible_rows + 1) * base_row_height
    
        grid_response = AgGrid(
                                results_df,
                                gridOptions=grid_options,
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                height=height,
                                fit_columns_on_grid_load=True
                              )
        
        # Download raw values
        rolling_reg_csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
                            label="Download Rolling Regression Results",
                            data=rolling_reg_csv,
                            file_name="rolling_regression_results.csv",
                            mime="text/csv",
                          )
        
    
        selected = grid_response["selected_rows"]
        
        # Prevent memory leak
        plt.close("all")
        
        if isinstance(selected, pd.DataFrame) and not selected.empty:
            sel_filename = selected["Filename"][0]
            sel_channel = selected["Channel"][0]
            start, end = selected["Start"][0], selected["End"][0]
            slope, intercept = selected["Slope"][0], selected["intercept"][0]

            # Create a cache key for this channel
            cache_key = f"rolling_trace_{sel_filename}_{sel_channel}"

            # Check if we already have the main channel trace cached
            if cache_key in st.session_state:
                channel_trace = st.session_state[cache_key]  # Use cached trace
            else:
                # Optional downsampling to speed up plotting
                x_sel, y_sel = final_x_y_valid_dict[sel_filename][sel_channel]
                max_points = 1000
                if len(x_sel) > max_points:
                    idx = np.linspace(0, len(x_sel)-1, max_points).astype(int)
                    x_plot, y_plot = x_sel[idx], y_sel[idx]
                else:
                    x_plot, y_plot = x_sel, y_sel

                # Build the main channel trace
                channel_trace = go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=sel_channel,
                    line=dict(color="blue", width=2),
                    opacity=0.8
                )
                st.session_state[cache_key] = channel_trace  # Cache the main trace for future use

            # Build figure dynamically for the selected window
            fig = go.Figure()
            fig.add_trace(channel_trace)  # Add main channel trace

            # Highlight regression line for selected window (dynamic every time)
            x_sel, y_sel = final_x_y_valid_dict[sel_filename][sel_channel]  # Full channel
            mask = (x_sel >= start) & (x_sel <= end)
            fig.add_trace(go.Scatter(
                                    x=x_sel[mask],
                                    y=slope * x_sel[mask] + intercept,
                                    mode="lines",
                                    # name=f"Fit {selected['Window']}",
                                    line=dict(color="black", width=3)
                                    ))

            # Shade regression region
            fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="black", opacity=0.1, line_width=0
                         )

            fig.update_layout(
                            title=f"{sel_filename} — Channel {sel_channel}",
                            xaxis=dict(
                                title="Time (s)",
                                title_font=dict(size=18, color="blue"),
                                tickfont=dict(size=14, color="blue"),
                                showline=True,
                                linecolor="black",
                                linewidth=2,
                                mirror=True
                            ),
                            yaxis=dict(
                                title="Signal",
                                title_font=dict(size=18, color="blue"),
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
                            height=400 + 100 * ((len(final_x_y_valid_dict[sel_filename]) - 1) // 2),
                            margin=dict(l=80, r=40, t=60, b=60)
                           )

            st.plotly_chart(fig, use_container_width=True)
            
    def download_regression(self, all_subset_regs, volume_entry_df):
        """Create a download button for regressions of the data after the user
        adjusts the x-axis."""
        
        rows = []
        for filename, channels in all_subset_regs.items():     
            for ch, results in channels.items():
                
                # Get volume and coral ID
                volume = float(volume_entry_df.loc[(volume_entry_df['Filename'] == filename)
                                                   & (volume_entry_df['Channel'] == ch), 
                                                   'Volume (mL)'].iloc[0])
                
                coral_id = str(volume_entry_df.loc[(volume_entry_df['Filename'] == filename)
                                                   & (volume_entry_df['Channel'] == ch), 
                                                   'Coral ID'].iloc[0])
                
                
                # Format slope 95% CI to display correctly in excel
                ci = results.get("slope 95% CI")
                slope_CI = f"'{ci}"
                rows.append({
                            "Filename": filename,
                            "Channel": ch,
                            "Coral ID": coral_id,
                            "Volume (mL)": volume,
                            "Start Time": results.get("Start Time", np.nan),
                            "Stop Time": results.get("Stop Time", np.nan),
                            "slope": results.get("slope", np.nan),
                            "slope 95% CI": slope_CI,
                            "R2": results.get("R2", np.nan),
                            "slope pval": results.get("slope pval", np.nan),
                            "slope stderr": results.get("slope stderr", np.nan),
                            "SSR": results.get("squared residuals", np.nan),
                            "Lags": results.get("Maxlags", np.nan),
                           })
    
        # Raw dataframe (for CSV)
        df_raw = pd.DataFrame(rows)
    
        # Formatted copy (for display only)
        df_display = df_raw.copy()
        
        df_display["slope"] = df_display["slope"].apply(
            lambda x: round(x, 4) if isinstance(x, (float, int)) else x
        )
        df_display["R2"] = df_display["R2"].apply(
            lambda x: round(x, 3) if isinstance(x, (float, int)) else x
        )
        df_display["SSR"] = df_display["SSR"].apply(
            lambda x: round(x, 2) if isinstance(x, (float, int)) else x
        )
        df_display["slope pval"] = df_display["slope pval"].apply(
            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) else x
        )
        df_display["slope stderr"] = df_display["slope stderr"].apply(
            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) else x
        )
    
        # Show formatted table
        st.markdown(
                    "<p style='color: Blue; font-size: 24px; margin: 0;'>All Adjusted Regression Results</p>",
                    unsafe_allow_html=True
                   )
        
        st.data_editor(df_display, use_container_width=True, num_rows="static", hide_index=True)
    
        # Download raw values
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
                            label="Download Regression Results",
                            data=csv,
                            file_name="regression_results.csv",
                            mime="text/csv",
                          )

        
    def download_respiration_plots(self, all_plots):
        """Downloads plots as pdf."""
        
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
                                label="Download Plots",
                                data=pdf_buffer,
                                file_name="plots.pdf",
                                mime="application/pdf"
                              )
     

    
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
    
    # Disable directory when deploying to cloud
    # directory = '/Users/danfeldheim/Documents/mote_o2_app/deployed_versions/'
        
    # Use this for cloud
    st.session_state['logo'] = 'mote_logo.png'
    
    # Use this for local machine
    # st.session_state['logo'] = directory + 'mote_logo.png'
        
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
