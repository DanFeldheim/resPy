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
import io
from styles import CSS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from scipy.stats import t
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
from st_aggrid.shared import ColumnsAutoSizeMode
import plotly.graph_objects as go
import gc
import psutil
import heapq
import re
from scipy.stats import gaussian_kde
from PIL import Image
import math
import hashlib


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
       
    def all_calls(self):
        """This is the main logic workflow. All calls to other functions are here."""
        
        #---------------------------------------------------------------------------------------------
        # Initialize session state variables and dicts
        state_variables_dict = {
                                "download_options":'Combined',
                                "noise_analysis":False,
                                "residuals_plot":False,
                                "acf_plot":False,
                                "newey":False,
                                "normal":False,
                                "rolling_regression":False,
                                "tab1_files":[],
                                "tab2_files":[], 
                                "warning":False,
                                "uploaded_files_hash":None,
                                "model_plots":False, 
                                "show_model_plots":False,
                                "diag_pdf_bytes":None,
                                "pending_download":False
                                }
        
        # Add to session_state
        Utilities.add_to_state(state_variables_dict)
        
        # Instantiate dicts for storing data as we go through the work flow
        # Holds raw data
        edited_raw_dfs_dict = {}
        # Holds data after adjusting x-axis limits
        final_x_y_valid_dict = {}
        # Holds the regression fit stats
        all_regression_results = {}
        # Holds plots for downloading
        all_plots = {}
        # Holds rolling regression plots
        rolling_reg_plots = {}
        
        # Master regression/residuals results dict
        # These are dicts that hold regressions and residuals for the channels
        # selected for plotting by clicking checkboxes in the aggrid table
        all_subset_regs = {}
        all_subset_residuals = {}
       
        # Show loading message for first-time initialization
        st.write('')
        st.write('')
        st.write('')
     
        # Set up the header, login form, and check user credentials
        # Create Setup instance
        setup = Setup()
        
        # This is a diagnostic for checking memory usage
        # nav_bar = setup.navigation()
        
        # Render the header
        setup.header()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = setup.tabs()

        #---------------------------------------------------------------------------------------------
        # Make calls to import, load, and analyze data
        
        with tab1:
            
            # Call Load_Data class
            load_files = Load_Data()
    
            # Create an upload button to load file names to be uploaded for the 
            # raw data and master physio metadata spreadsheet
            photo_files, master_file = load_files.upload()
            
            st.divider()
            
            if master_file and photo_files:
                
                # Loop through files and generate a dictionary of cleaned up x-y data dataframes
                # with filenames as keys and data, run#, group#, etc as values
                for file in photo_files:
                    
                    # Process the data; returns dataframe of raw channel data without NaNs
                    photo_filename, date, start_time, edited_df = self.preprocess_data(file, 
                                                                                       load_files, 
                                                                                       )
                    
                    # Remove spaces from the edges
                    photo_filename = photo_filename.strip()
                    
                    # Remove spaces next to an underscore
                    photo_filename = re.sub(r"\s*_\s*", "_", photo_filename)
                    # st.write(photo_filename)
                  
                    try: 
                        # Extract run #, group, and light_dark from filename (e.g., Run1B, Run15A)
                        # The re function tolerates a lot of variations in the filename entered
                        match = re.search(
                                          r"(?<![A-Za-z0-9])run[\s_-]*0*(\d+)[\s_-]*([A-Za-z]+)(?![A-Za-z0-9])",
                                          photo_filename,
                                          re.IGNORECASE
                                         )
                        
                        # Extract run and group numbers
                        if match:
                            run = int(match.group(1))          
                            group = match.group(2).upper()   
                          
                        else:
                            run = None
                            group = None
                        
                        # Extract light or dark
                        light_dark = "light" if "light" in photo_filename.lower() else "dark"
                        
                        # Add the final raw edited_df and metadata extracted from filename to a dict 
                        edited_raw_dfs_dict[photo_filename] = {"data": edited_df,
                                                               "date": date,
                                                               "start_time":start_time,
                                                               "run":run,
                                                               "group":group,
                                                               "light_dark":light_dark}
                        
                        # If regex didn’t match, issue a targeted warning
                        if run is None: 
                            st.warning(
                                        f"⚠️ Could not extract Run Number from filename '{photo_filename}'. "
                                        "Check filename format (e.g., 08_01_25_Run_1A_dark)."  
                                      )
                            
                    except Exception as e:
                        
                        st.warning(f"⚠️ Error parsing filename '{photo_filename}': {e}")
                        
                        # Still record the file with None values so you can track it
                        edited_raw_dfs_dict[photo_filename] = {  
                                                                "data": edited_df,
                                                                "date": date,
                                                                "start_time": start_time,
                                                                "run": None,
                                                                "group": None,
                                                                "light_dark": None
                                                              }
                        continue 
                    
              
                # Get the metadata file
                master_file_df = load_files.upload_master(master_file)
               
                # Match run, run letter, and light_dark to the physiomaster spreadsheet.
                # to extract volume and coral ID from the physiomaster file.
                # Debug will activate a series of error messages for missing data.
                edited_raw_dfs_dict = load_files.find_volume(edited_raw_dfs_dict, master_file_df) 
            
                #--------------------------------------------------------------------------------------
                # Congratulations! You now have a dict with channel data, volume, and coral ID
                    
                # Loop through edited_raw_dfs_dict and do
                # linear regression to get slope, R2, p-value, sum of squared residuals,
                # slope standard error, and 95% CI
                # Create an aggrid table for results
                
                # Check for data
                if len(edited_raw_dfs_dict) > 0:
                   
                    # Call the Analysis class to plot data
                    plot_data = Analysis()  
                    
                    # Filename here is the key, data is val
                    for filename, data in edited_raw_dfs_dict.items():
                        
                        # Get data and metadata
                        df = data['data']
                        channel_metadata = data['channel_metadata']
                        
                        # Only keep valid channel columns
                        channels = [c for c in df.columns if c.startswith("Ch") and df[c].notna().any()]
                        
                        # Loop through channels
                        for ch in channels:
                            x = df['Time (s)'].values
                            start_time = float(np.min(x))
                            stop_time = float(np.max(x))
                            duration = stop_time - start_time
                            
                            # Apply pressure correction of 1013/1013.25 = 0.99975
                            # Also apply volume correction = vol/1000
                            try:
                                
                                # Get volume to do volume correction
                                volume = channel_metadata[ch]["volume_mL"]
                                coral_id = channel_metadata[ch]["coral_ID"]
                            
                                # Apply corrections directly to the DataFrame column so that
                                # edited_raw_dfs_dict is changed at all points downstream
                                # df[ch] = df[ch] * 0.99975 * (volume / 1000)
                                
                                # Use this for no pressure correction-Requested by Emily
                                df[ch] = df[ch] * (volume / 1000)
                                
                                # Extract volume corrected y values for regression
                                y = df[ch].values
                    
                                # Fit regression
                                regress, residuals = plot_data.fit_regression(x, y)
                                
                                # Calculate various noise params using helper function
                                rmse, noise_percent, rse_percent, zmax = self.noise_calc(residuals, 
                                                                                         regress, 
                                                                                         duration)
                                
                                # Scan the selected interval for substantially different sustained
                                # 5-minute respiration rates.
                                stability_results = plot_data.rate_shift_analysis(x, y)
                    
                                # Estimate maxlags as an indication of residual autocorrelation
                                # The max lag returned is based upon the 95% CI of ACF
                                maxlag = plot_data.estimate_maxlags(residuals)
                    
                                # Store results for all channels and filenames in one dict
                                all_regression_results.setdefault(filename, {})[ch] = {
                                                                                        # Unpack regression stats
                                                                                        **regress,                 
                                                                                        "Start Time": start_time,
                                                                                        "Stop Time": stop_time,
                                                                                        "System Lag (Pts)": maxlag, 
                                                                                        "Volume (mL)":volume,
                                                                                        "Coral ID":coral_id,
                                                                                        "RMSE":rmse,
                                                                                        "Noise %":noise_percent,
                                                                                        "slope % RSE":rse_percent,
                                                                                        "Spike Score":zmax,
                                                                                        **stability_results
                                                                                        }
                                
                            except Exception as e:

                                # Preserve metadata status returned by find_volume()
                                volume = channel_metadata[ch].get("volume_mL")
                                coral_id = channel_metadata[ch].get("coral_ID")
                            
                                # Use display placeholders only when no value/status was supplied
                                if volume is None:
                                    volume = "Not found"
                            
                                if coral_id is None:
                                    coral_id = "Not found"
                            
                                all_regression_results.setdefault(filename, {})[ch] = {
                                                                                        "Error": e,
                                                                                        "Start Time": start_time,
                                                                                        "Stop Time": stop_time,
                                                                                        "System Lag (Pts)": None,
                                                                                        "Volume (mL)": volume,
                                                                                        "Coral ID": coral_id,
                                                                                        "RMSE": None,
                                                                                        "Noise %": None,
                                                                                        "slope % RSE": None,
                                                                                        "Spike Score": None
                                                                                      }
                            
                                continue
                
                st.write("")
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>Preview Linear Regression Results</p>",
                      unsafe_allow_html=True)
                
                # Show results in aggrid table and return selected rows
                selected_rows = plot_data.agtable(all_regression_results, grid_key = "results_preview")
              
                st.markdown(f"""
                            <p style='color: DarkRed; 
                                      font-size: 20px; 
                                      margin: 0;'>
                                      Select channels to view and adjust plots.<br>
                                      Then scroll to the bottom to download adjusted results table.
                            </p>
                            """, unsafe_allow_html=True)
                    
                st.divider()
                st.write('')
            
                #--------------------------------------------------------
                # Now allow the user to select channels for further data processing and analysis
                # Selecting channels triggers plots with slider bars to crop the data
                # Then a new table of regression data are rendered under each plot
                
                # Subset data based upon selected rows
                subset_data = plot_data.subset_dataframe(selected_rows, edited_raw_dfs_dict)
                
                # Pass subset_data df to slider_bars() and plot functions
                # Slider bars allow user to change the x-axis and crop out spikes in the middle of a run
                if subset_data is not None:
                    
                    for filename, vals in subset_data.items():
                        
                        df = vals['data']

                        st.markdown(f"<p style='color: Black; \
                                      font-size: 24px; \
                                      margin: 0;'>{filename}</p>",
                                      unsafe_allow_html=True)
                    
                        # edited_df holds the data after user has changed the x-axis
                        # Here it is passed to a function that creates slider bars
                        # for each channel of each file to change x-axis
                        edited_df, crop_info = plot_data.slider_bars(df, filename, tab1)
                        
                        # Plots and Returns new x and y values as a dict with
                        # channel as key and x_valid, y_valid as a list of 
                        # 1D arrays. e.g. ch1:[[x_valid], [y_valid]]
                        x_y_valid_dict, all_plots = plot_data.plot_channels(edited_df, 
                                                                            filename, 
                                                                            all_plots,
                                                                            crop_info=crop_info)
                        
                        # Store all x/y values for all filenames and channels
                        # after changing x-axis (or not)
                        final_x_y_valid_dict[filename] = x_y_valid_dict
                        
                        # Container for this file's channel regression results
                        channel_results = {}
                        
                        # Container for this file's channel residual results
                        channel_residuals = {}
                       
                        # Do the regression on the edited data
                        for channel, (x_valid, y_valid) in x_y_valid_dict.items():
                            
                            # Get the start/stop times for the x-axis
                            start_time = float(min(x_valid))
                            stop_time = float(max(x_valid))
                            duration = stop_time - start_time
                            n_points  = x_valid.size
                           
                            # Do the regression and get maxlags
                            regress, residuals = plot_data.fit_regression(x_valid, y_valid)
                            
                            # Get the effective # of points
                            # Add residuals to a dict and pass to ESS function, which requires a dict
                            unique_pts = plot_data.ESS_correction_from_residuals(residuals)
                            
                            # Calculate noise % and other noise parameters
                            rmse, noise_percent, rse_percent, zmax = self.noise_calc(residuals, 
                                                                                     regress, 
                                                                                     duration)
                            
                            # Scan the selected interval for a sustained long-window rate shift.
                            stability_results = plot_data.rate_shift_analysis(x_valid, y_valid)
                            
                            # Get per-channel metadata. If none exist return empty dict
                            metadata = vals['metadata'].get(channel, {})
                            coral_id = metadata.get('coral_id', 'Unknown')
                            volume = metadata.get('volume', 'Unknown')
                            treatment = metadata.get('treatment', 'Unknown')
                            
                            # Do the Newey-West correction
                            maxlag = plot_data.estimate_maxlags(residuals)
                            nw_results = plot_data.newey_west_analysis(x_valid, y_valid, lags=maxlag)
                            
                            # Combine results for the current channel in a dict
                            channel_results[channel] = {
                                                        **regress,
                                                        **nw_results,
                                                        "System Lag (Pts)": maxlag,
                                                        "Start Time":start_time,   
                                                        "Stop Time":stop_time,
                                                        "Total Pts":n_points,
                                                        "Unique Pts":unique_pts,
                                                        "Volume (mL)":volume,
                                                        "Coral ID":coral_id,
                                                        "Noise %": noise_percent,
                                                        "slope % RSE":rse_percent,
                                                        "RMSE":rmse,
                                                        "Spike Score":zmax,
                                                        "Treatment":treatment,
                                                        **stability_results
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
                                                               use_checkboxes = False,
                                                               show_warning = False,
                                                               grid_key = f"{tab1}_processed_results_{filename}"
                                                              )
                        
                        # Subtract the blank slope from each channel slope
                        all_subset_regs = plot_data.blank_subtraction(all_subset_regs)
                        
                        st.divider()
                    
                    try:
                        
                        # Displays a table of all regression results for selected channels
                        # and a button to download regression results.
                        download_reg = plot_data.download_regression(all_subset_regs)   
                    
                        # Download plots
                        download_plots = plot_data.download_respiration_plots(all_plots)
                        
                        # Add a dropdown explaining the results table
                        col1, col2 = st.columns([1,1])
                        with col1:
                            
                            st.write('')
                            About().results_interpretation()
                            
                        st.divider()
                    
                    except Exception as e:
                        st.write(e)
                        
              
        # Model Diagnostics calls
        #---------------------------------------------------------------------------------------------    
        with tab2:    
            
            st.write('')
  
            # If user selected files in the Linear Regression tab
            # then all_subset_regs will exist and we can continue
            if len(all_subset_regs) > 0:
              
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
            
                # Button columns
                col1, col2, col3, spacer = st.columns([1, 1, 1.4, 6], gap="small")
                
                # Change session_state when buttons are clicked
                with col1:
                    if st.button("View Plots", key="btn_render"):
                        st.session_state.show_model_plots = True
                        st.session_state.pending_download = False
                        st.rerun()
                        
                    st.write('')
                    if st.button("Build PDF", key="btn_download"):
                        st.session_state.pending_download = True
                        st.session_state.show_model_plots = False  
                        st.rerun()
                
                with col2:
                    if st.button("Clear Plots", key="btn_clear"):
                        st.session_state.show_model_plots = False
                        st.session_state.pending_download = False
                        st.rerun()  
                        
                    st.write('')
                        
                    # Create an area for Building PDF message
                    download_area = st.container()
                    
                    if st.session_state.pending_download:
                        
                        # Build PDF bytes once
                        with download_area:
                            
                            with st.spinner("Building PDF..."):
                                st.session_state.diag_pdf_bytes = plot_data.build_pdf(
                                                                                      all_subset_residuals,
                                                                                      max_lag=200,
                                                                                      hard_cap=50,
                                                                                      dpi=120,
                                                                                     )
                    
                            # After building, show the download button
                            st.download_button(
                                              "Export PDF",
                                              data=st.session_state.diag_pdf_bytes,
                                              file_name="diagnostic_plots.pdf",
                                              mime="application/pdf",
                                              key="btn_download_ready",
                                              )
                    
                            # Reset the pending flag so we don't rebuild on every rerun
                            st.session_state.pending_download = False
                
                # Create a container for the plots
                plots_area = st.container()
                if st.session_state.show_model_plots:
                    with plots_area:
                        st.divider()
                        plot_data.render_plots(all_subset_residuals, max_lag=200, hard_cap=50)    
                        
            else:
                
                col1, col2, col3 = st.columns([0.5,6,1])
                
                with col2:
                
                    st.markdown(f"<p style='color: Red; \
                          font-size: 24px; \
                          margin: 0;'>Please go to the Linear Regression tab, upload data, and select channels to analyze.</p>",
                          unsafe_allow_html=True)
                        
            # Rolling regression and most common slope calls
            #--------------------------------------------------------------------------------------------- 
            with tab3:    
                
                st.write('')
      
                # If user selected files in the Linear Regression tab
                # then all_subset_regs will exist and we can continue
                if len(all_subset_regs) > 0:
                    
                    col1, col2 = st.columns([1,1])
                    
                    # Slider bar to set window size
                    with col1:
                        
                        window_seconds = st.slider(
                                                    "Select rolling regression window length (seconds).",
                                                    min_value=300,
                                                    max_value=900,
                                                    value=600,
                                                    step=30,
                                                    key=f"win_size_{filename}"
                                                  )
                    st.write('')
                    
                    st.markdown(f"<p style='color: dodgerblue; \
                                  font-size: 20px; \
                                  margin: 0;'>Sliding Windows Ranked by Closest Match to Most Common Slope</p>",
                                  unsafe_allow_html=True)
                        
                    st.write('')

                    st.markdown(f"<p style='color: black; \
                                  font-size: 14px; \
                                  margin: 0;'>Select a channel to view plot and fit line.</p>",
                                  unsafe_allow_html=True)
                    
                    st.write('')
                       
                    # Computes regression fits for small windows in the data
                    rolling_reg = plot_data.compute_rolling_regression(
                                                                        final_x_y_valid_dict,
                                                                        all_subset_regs,
                                                                        window_seconds
                                                                      )
                    
                    # Displays rolling regression table and plot
                    rolling_reg_plots = plot_data.rolling_reg_ui(final_x_y_valid_dict, rolling_reg)
                    
                    # Calculates the most common slope in the data using the rolling_reg slopes.
                    # Uses a kernal density estimate (KDE) algorithm to fit a gaussian to the 
                    # slope data to find the most common slope
                    kde_analysis_df = plot_data.kde(rolling_reg, slope_col="Raw slope (umol/hr)")
                    
                    # Render the results
                    plot_data.kde_render(kde_analysis_df)

                else:
                    
                    col1, col2, col3 = st.columns([0.5,6,1])
                    
                    with col2:
                    
                        st.markdown(f"<p style='color: Red; \
                              font-size: 24px; \
                              margin: 0;'>Please go to the Linear Regression tab, upload data, and select channels to analyze.</p>",
                              unsafe_allow_html=True)
                            
            with tab4:
                
                slides = About().about_tab()
                            
                            
                
    #------------------------------------------------------------------------------------------------------------
    # Some "helper" functions to the flow control class
    def preprocess_data(self, file, load_files):
        """Helper function to the flow control class to clean up the uploaded data before calculations."""

        # Get file name
        file_name = file.name
       
        # Strip out .csv from file name
        filename, _ = os.path.splitext(file_name)
        
        # Import into dataframe. Calls function that caches the data
        import_data = load_files.import_df(file)
        
        # Grab date, start_time and oxygen concentration columns
        date, start_time, oxy_df = load_files.oxy_data(import_data)
        
        # Convert everything to numeric (non-numeric → NaN)
        oxy_numeric = oxy_df.apply(pd.to_numeric, errors='coerce')
        
        # Drop columns where all values are NaN
        oxy_column_cleaned = oxy_numeric.dropna(axis=1, how='all')
        
        # Drop rows where any column is NaN
        edited_df = oxy_column_cleaned.dropna(axis=0, how='any')
    

        return filename, date, start_time, edited_df
    
    def about_model_diagnostics(self):
        """A helper function to the flow control class that explains the various model diagnostics."""
        
        with st.expander('🪸 About Model Diagnostics'):
            
            st.markdown(
                        """
                        <div style="text-align: left; line-height: 1.6;">
                        <h3>🎛️ Noise Analysis</h3>
                        <ul>
                        <li><b>Residuals Plots</b> → Residuals should be scattered randomly around the 0 line.<br>
                        A non-random pattern indicates model assumptions may be violated, such as autocorrelation, 
                        heteroskedasticity, or nonlinearity.<br>
                        
                        In time-series experiments, autocorrelation often arises because the system has <b>memory</b> 
                        (e.g., slow mixing or sensor response).<br>
                    
                        A key consequence is that standard errors, confidence intervals, and p-values are underestimated. 
                        These can be corrected using methods such as Newey–West.<br>
                    
                        <li><b>Autocorrelation Function (ACF) Plots</b> → Shows how similar each measurement is to earlier 
                        measurements.<br>
                        Lag = how far back points are compared in time.<br> 
                        y-axis values range from -1 (negatively correlated) to +1 (positively correlated).<br><br>
                        If nearby points are similar, the system has <b>memory</b>.<br>
                    
                        <b>Interpretation:</b><br>
                        • <b>Slow, smooth decrease in height:</b> Substantial memory in the system → slope often accurate, 
                        but SE underestimated. Use Newey-West Corrected SE for inferential statistics<br>
                        • <b>Oscillating bars:</b> warning! → slope may not be accurate<br>
                        • <b>Near zero:</b> independent data → standard statistics valid
                    
                        <li><b>Normality Tests</b> → Residuals should be roughly normally distributed.<br>
                        Major deviations suggest outliers or model issues.</li>
                        </ul>
                        <hr>
                        <h3>📈 Signal Processing</h3>
                        <ul>
                        <li><b>Correlation Correction</b> → When residuals are correlated, error bars and p-values 
                        become too optimistic.<br><br>
                    
                        Newey–West corrects this by widening uncertainty estimates like SE and 95% CI.<br>
                        Effective Sample Size (ESS) is the number of independent data points.<br>
                    
                        ESS is best used as a diagnostic (small ESS = sluggish system), while Newey–West provides 
                        corrected statistics for reporting.</li>
                        </ul>
                        </div>
                        """,
                            unsafe_allow_html=True
                        )
            
    def noise_calc(self, residuals, regress, duration_s):
        """Compute RMSE, Noise %, RSE %, and Zmax for a fitted line window."""
        
        # Get residuals and rmse
        res = np.asarray(residuals, dtype=float)
        rmse = float(np.sqrt(np.mean(res ** 2))) if res.size else np.nan
    
        # Defaults in case of failure
        noise_percent = np.nan
        rse_percent = np.nan
        zmax = np.nan
    
        # Get slope and stderr
        slope = regress.get("slope (umol/hr)", np.nan)
        stderr = regress.get("slope stderr (umol/hr)", np.nan)
    
        # Zmax: spike/outlier score (largest residual relative to typical residual scale)
        if res.size and np.isfinite(rmse) and rmse > 0:
            zmax = float(np.max(np.abs(res)) / rmse)
    
        # If slope isn't valid, we can't compute Noise% or RSE%
        if not isinstance(slope, (float, int)) or not np.isfinite(slope):
            return rmse, noise_percent, rse_percent, zmax
    
        duration_hr = float(duration_s) / 3600.0

        # Noise % = RMSE / total signal change across the window
        delta_y = abs(slope) * duration_hr
        
        if np.isfinite(delta_y) and delta_y > 0:
            noise_percent = 100.0 * rmse / delta_y if np.isfinite(rmse) else np.nan
    
        # RSE % = slope stderr / |slope|
        if isinstance(stderr, (float, int)) and np.isfinite(stderr) and abs(slope) > 0:
            rse_percent = 100.0 * stderr / abs(slope)
            
 
        return rmse, noise_percent, rse_percent, zmax
        
    
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
        tab1, tab2, tab3, tab4 = st.tabs(["Linear Regression", "Model Diagnostics", "Rolling Regression", 'About ResPy'])
        
        return tab1, tab2, tab3, tab4

        
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
          
class Load_Data():
    """Creates a file uploader and imports data as dataframe."""
    
    def __init__(self):
        
        pass
    
    def upload(self):
        
        col1, col2 = st.columns([1,1])
        
        with col1:
        
            # File uploader that allows multiple files
            uploaded_files = st.file_uploader(
                                              "Choose Photosynthesis Data Files", 
                                              type=["csv", "txt"],  
                                              accept_multiple_files=True, 
                                              key="photo_files_uploader"
                                              )
            
        with col2: 
            
            # File uploader that allows a single file
            master_file = st.file_uploader(
                                           "Choose Physio Master File", 
                                           type=["csv", "txt"],  
                                           accept_multiple_files=False, 
                                           key="master_file_uploader"
                                           )
            
        return uploaded_files, master_file
        
    @staticmethod    
    @st.cache_data(max_entries=40, ttl=14400)
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
                    file.seek(0)
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
    
    def upload_master(self, file):
        
        file.seek(0)
        
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        
        for enc in encodings_to_try:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                
                return df
            
            except UnicodeDecodeError:
                continue
            
            except Exception as e:
                st.error(f"❌ Error reading CSV file {file.name}: {e}")
                return None
        
    @staticmethod
    @st.cache_data(max_entries=40, ttl=14400)
    def oxy_data(df):
        """Retrieves oxygen channels and time."""
        
        # Get the start time of the experiment
        date = df.iloc[0,0]
        start_time = df.iloc[0,1]
        start_time = start_time[0:-3]
        # Remove leading 0 on start times earlier than 10:00. 
        start_time = start_time[1:] if start_time.startswith("0") else start_time
       
        # Get date, time in Time (HH:MM:SS), time in sec, and O2 concentrations
        oxy_df = df.iloc[:, [2, 4, 5, 6, 7]]
        
        # Time in seconds 
        oxy_df.columns = ['Time (s)', 'Ch1', 'Ch2', 'Ch3', 'Ch4']
      
        return date, start_time, oxy_df
    
    def find_volume(self, edited_raw_dfs_dict, master_file_df):
        """
        Takes in data from edited_raw_dfs_dict and retrieves volume and coral ID
        from master_file_df (the metadata file). Adds these to the
        edited_raw_dfs_dict and returns it.
        """
    
        # Copy for safety
        mdf = master_file_df.copy()
       
        # Normalize and validate user-entered dates in the physio master file
        mdf["date"] = (
                        mdf["date"]
                        .astype(str)
                        .str.strip()
                        .str.replace(r"[_\.-]", "/", regex=True)
                        .str.replace(r"/+", "/", regex=True)
                      )
    
        parsed_dates = pd.to_datetime(
                                      mdf["date"],
                                      errors="coerce",
                                      format="mixed"
                                     )
    
        bad_date_rows = parsed_dates.isna()
    
        if bad_date_rows.any():
            st.error(
                     "One or more dates in the physio master file could not be interpreted. "
                     "Please correct the highlighted rows and reload the file."
                    )
    
            st.dataframe(
                         master_file_df.loc[
                                            bad_date_rows,
                                            [
                                                "date",
                                                "filename",
                                                "run",
                                                "channel",
                                                "group",
                                                "light_dark"
                                            ]
                                            ]
                        )
    
            return None
    
        mdf["date"] = parsed_dates.dt.date
    
        # Normalize remaining metadata columns
        mdf["filename"] = (
                            mdf["filename"]
                            .astype(str)
                            .str.strip()
                            .apply(lambda x: os.path.splitext(x)[0])
                            .str.lower()
                          )
    
        mdf["run"] = pd.to_numeric(
                                   mdf["run"],
                                   errors="coerce"
                                  ).astype("Int64")
                        
        mdf["channel"] = (
                          mdf["channel"]
                          .astype(str)
                          .str.strip()
                          .str.lower()
                         )
    
        mdf["group"] = (
                        mdf["group"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                       )
    
        mdf["light_dark"] = (
                             mdf["light_dark"]
                             .astype(str)
                             .str.strip()
                             .str.upper()
                            )
    
        # Loop through data files and match data with metadata
        for clean_filename, file_info in edited_raw_dfs_dict.items():
    
            # Normalize information parsed from the respiration filename
            filename = os.path.splitext(str(clean_filename).strip())[0].lower()
            file_date = pd.to_datetime(str(file_info["date"]).strip(),errors="coerce").date()
            file_run = pd.to_numeric(str(file_info["run"]).strip(),errors="coerce")
            file_run = int(file_run) if pd.notna(file_run) else None
            file_group = (str(file_info["group"]).strip().upper())
            file_light_dark = (str(file_info["light_dark"]).strip().upper())

            # Retrieve dataframe of time and channel data
            df = file_info["data"]
            
            # Start a temp dict for channel metadata
            channel_metadata = {}
    
            # Find metadata rows matching the experimental identifiers
            # other than filename and channel
            # Creates a new boolean column of match == True or False and returns
            # that along with all the cols in mdf
            run_subset = mdf[
                             (mdf["date"] == file_date) &
                             (mdf["run"] == file_run) &
                             (mdf["group"] == file_group) &
                             (mdf["light_dark"] == file_light_dark)
                            ]
    
            # Match metadata separately for each channel
            for ch in (c for c in df.columns if str(c).lower().startswith("ch")):
    
                # Convert all channel columns to lowercase
                channel = (str(ch).strip().lower())
    
                # First determine whether metadata exists for this
                # date/run/group/light-dark/channel combination
                channel_subset = run_subset[run_subset["channel"] == channel]
    
                # If no metadata row exists for this channel
                if channel_subset.empty:
                    channel_metadata[ch] = {
                                            "volume_mL": None,
                                            "coral_ID": "Not Found",
                                            "treatment": None,
                                           }
                    # Move to next channel
                    continue
    
                # If metadata exists for this channel.
                # Now require the filename to match as well.
                match = channel_subset[channel_subset["filename"] == filename]
    
                # Experimental identifiers match, but filename does not
                if match.empty:
                    channel_metadata[ch] = {
                                            "volume_mL": "Mismatched Filename",
                                            "coral_ID": "Mismatched Filename",
                                            "treatment": None,
                                           }
    
                # Complete metadata match
                else:
                    channel_metadata[ch] = {
                                            "volume_mL": match["volume_mL"].iloc[0],
                                            "coral_ID": match["coral_ID"].iloc[0],
                                            "treatment": match["treatment"].iloc[0],
                                           }
    
            edited_raw_dfs_dict[clean_filename]["channel_metadata"] = channel_metadata
    
        return edited_raw_dfs_dict
            
    
    
class Analysis():
    
    """Performs all calculations-linear regression, noise, 
    and model analysis-and generates plots, tables, and download buttons."""
    
    # Long-window rate-shift settings. Each diagnostic fit spans 5 minutes.
    # Consecutive windows overlap by only 20%, so a new window starts every
    # 4 minutes. The final possible 5-minute window is always included.
    RATE_SHIFT_WINDOW_SECONDS = 300.0
    RATE_SHIFT_OVERLAP_PERC = 20.0

    # Rate-shift thresholds. A pair of adjacent 5-minute windows is only
    # analyzed when at least one of the two slopes reaches the minimum
    # meaningful rate. This prevents nearly flat/noisy windows from creating
    # large percentage changes that are not biologically useful. Eligible
    # pairs are compared using relative slope difference only.
    # These values are intentionally centralized so they can be calibrated
    # against known Mote traces as more runs are reviewed.
    RATE_SHIFT_MIN_SLOPE = 1.0                 # umol/hr
    RATE_SHIFT_PERCENT_THRESHOLD = 75.0        # percent

    # Rolling-regression summary settings. The dominant KDE peak is used as
    # the most common slope. Rate variability describes the spread of all
    # rolling slopes relative to that slope. Relative variability is not
    # calculated when the most common slope is near zero because division by
    # a small slope makes the percentage unstable and difficult to interpret.
    # These thresholds are intentionally centralized so they can be calibrated
    # against known Mote traces as more runs are reviewed.
    ROLLING_MIN_SLOPE = 1.0                       # umol/hr
    ROLLING_MIN_WINDOWS = 10                      # Minimum windows allowed 
    ROLLING_STABLE_VARIABILITY_PERCENT = 10.0     # percent
    ROLLING_HIGH_VARIABILITY_PERCENT = 50.0       # percent
    
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
    
    def agtable(self, 
                all_regression_results, 
                base_row_height: int = 30, 
                max_rows: int = 20, 
                use_checkboxes: bool = True,
                show_warning: bool = True,
                grid_key = "aggrid_table"):
        
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
                    
                maxlag = results.get("System Lag (Pts)")
                slope = results.get("slope (umol/hr)")
                lower_ci = results.get("CI Low (umol/hr)")
                upper_ci = results.get("CI High (umol/hr)")
                slope_ci = results.get("slope 95% CI (umol/hr)")
                R2 = results.get("R2")
                SSR = results.get("squared residuals")
                rmse = results.get("RMSE")
                pval = results.get("slope pval")
                stderr = results.get("slope stderr (umol/hr)")
                coral_id = results.get("Coral ID")
                vol = results.get("Volume (mL)")
                noise_percent = results.get("Noise %")
                rse_percent = results.get("slope % RSE")
                zmax = results.get("Spike Score")
                rate_shift = results.get("Rate Shift")
                total_pts = results.get("Total Pts")
 
                # Round numeric values
                slope = round(slope, 3) if isinstance(slope, (float,int)) else np.nan
                R2 = round(R2, 3) if isinstance(R2, (float,int)) else np.nan
                SSR = round(SSR, 2) if isinstance(SSR, (float,int)) else np.nan
                rmse = round(rmse, 3) if isinstance(rmse, (float, int)) else np.nan
                noise_percent = round(noise_percent, 1) if isinstance(noise_percent, (float, int)) else np.nan
                maxlag = maxlag if isinstance(maxlag, (float,int)) else np.nan
                rse_percent = round(rse_percent, 2) if isinstance(rse_percent, (float,int)) else np.nan
                zmax = round(zmax, 2) if isinstance(zmax, (float,int)) else np.nan
                rate_shift = rate_shift if rate_shift in ("Yes", "No", "Low Overall Rate") else ""
                lower_ci = round(lower_ci, 3) if isinstance(lower_ci, (float, int)) else np.nan
                upper_ci = round(upper_ci, 3) if isinstance(upper_ci, (float, int)) else np.nan
    
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
                        "Slope (umol/hr)": slope,
                        "R2": R2,
                        "Slope 95% CI (umol/hr)": slope_ci,
                        "Slope % RSE":rse_percent,
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        # "slope pval": pval_str,
                        # "p-value_numeric": pval if isinstance(pval,(float,int)) else np.nan,
                        # "slope stderr (umol/hr)": stderr_str,
                        # "SSR": SSR,
                        # "RMSE":rmse,
                        "Noise %":noise_percent,
                        "Spike Score":zmax,
                        "Rate Shift": rate_shift,
                        "System Lag (Pts)": maxlag
                      }
                
                # Append to rows list
                rows.append(row)
                
        # Build a dataframe from the results
        df = pd.DataFrame(rows)
        
        if df.empty:
            return []
        
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
        # Keep the rate-variation threshold tied to the class-level setting.
        cell_style_code = """
                                    function(params) 
                                        {
                                        if(params.value != null && params.colDef.field == 'R2' && params.value < 0.90) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'Noise %' && params.value > 25) {
                                        return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'Slope % RSE' && params.value > 25.0) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'Spike Score' && params.value > 4.0) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.colDef.field == 'Rate Shift' && params.value == 'Yes') {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if (
                                            params.colDef.field == 'Slope 95% CI (umol/hr)' &&
                                            params.data != null &&
                                            params.data.lower_ci != null &&
                                            params.data.upper_ci != null &&
                                            params.data.lower_ci <= 0 &&
                                            params.data.upper_ci >= 0
                                           ) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                            
                                        return null;
                                    };
                                    """
        cell_style_jscode = JsCode(cell_style_code)
    
        # Apply JS styling to each column that needs it
        # highlight_columns = ["slope pval", "R2", "SSR", "MaxLag"]
        highlight_columns = ["R2", "Noise %", "Slope % RSE", "Spike Score", "Rate Shift", "Slope 95% CI (umol/hr)"]
        
        for col in highlight_columns:
            gb.configure_column(col, cellStyle=cell_style_jscode)
    
        # Hide the helper numeric column
        # gb.configure_column("p-value_numeric", hide=True)
        gb.configure_column("lower_ci", hide=True)
        gb.configure_column("upper_ci", hide=True)

        # Narrow certain columns
        gb.configure_column(
                            "Slope 95% CI (umol/hr)",
                            width=150,
                            minWidth=130,
                            maxWidth=175
                           )
                    
        gb.configure_column(
                            "Slope % RSE",
                            width=95,
                            minWidth=85,
                            maxWidth=110
                           )
                    
        gb.configure_column(
                            "Noise %",
                            width=95,
                            minWidth=85,
                            maxWidth=110
                           )
        
        gb.configure_column(
                            "Spike Score",
                            width=105,
                            minWidth=95,
                            maxWidth=120
                           )
        
        gb.configure_column(
                            "Rate Shift",
                            width=120,
                            minWidth=110,
                            maxWidth=135
                           )
        
        gb.configure_column(
                            "System Lag (Pts)",
                            width=125,
                            minWidth=110,
                            maxWidth=140
                           )
                    
        grid_options = gb.build()
        
        results_key = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:8]
        
        grid_response = AgGrid(
                                df,
                                gridOptions=grid_options,
                                allow_unsafe_jscode=True,
                                # fit_columns_on_grid_load=True,
                                height=height,
                                key=f"{grid_key}_{results_key}",
                                update_on=["selectionChanged"]
                              )
                    
        # Show MaxLag warning in a popup window if needed
        # Controlled by session_state['warning'] and ['warning_rendered'] so that it only runs once
        # per upload and only after the first aggrid table
        if not st.session_state["warning"] and (df["System Lag (Pts)"] > 50).any():
   
            col1, col2, col3 = st.columns([1, 3, 1])

            if show_warning:
                
                # Use a placeholder to control exact layout
                placeholder = col2.empty()  
                
                with placeholder.container():
                    st.error(
                            "System Lag Warning! Runs with System Lag > 50 often reflect poor mixing "
                            "(e.g., bubbles, blocked flow).\n"
                            "This can cause underestimated standard errors and overly narrow 95% confidence intervals, "
                            "making results appear more precise than they are.\n"
                            "Use the lag-adjusted SE and lag-adjusted 95% CI in the Final Results Table for error analysis."
                            )
                        
                    # Button click sets warning to True
                    if st.button("Ok", key="maxlag_warning"):
                        st.session_state["warning"] = True
                        placeholder.empty()  
        
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
            
        if selected_rows is not None and not selected_rows.empty:
            # Drop rows where both Volume and Coral ID are "Not found" (or NaN)
            selected_rows = selected_rows[
                                           ~(
                                            (selected_rows["Volume (mL)"].fillna("Not found") == "Not found") &
                                            (selected_rows["Coral ID"].fillna("Not found") == "Not found")
                                            )
                                         ]
            
            # Loop through selected_rows df
            for _, row in selected_rows.iterrows():
                
                # Get filename and channel
                filename = row["Filename"]
                channel = row["Channel"]
                coral_id = row["Coral ID"]
                volume = row["Volume (mL)"]
                treatment = edited_raw_dfs_dict[filename]["channel_metadata"][channel].get("treatment", "Unknown")
    
                # Get the full dataframe for this file
                df = edited_raw_dfs_dict[filename]["data"]
                
                # Get the channel for this filename
                if df is not None and channel in df.columns:
                    
                    # Subset time + this channel
                    subset_df = df[["Time (s)", channel]].dropna()
                   
                    # If filename not seen yet, create a key with the filename in subset_data,
                    # otherwise merge the channel with the existing filename in subset_data.
                    # Initialize the filename entry if it doesn't exist
                    if filename not in subset_data:
                        subset_data[filename] = {
                                                "data": subset_df.rename(columns={channel: channel}),
                                                "metadata": {
                                                             channel: {
                                                                       "coral_id": coral_id,
                                                                       "volume": volume,
                                                                       "treatment": treatment,
                                                                      }
                                                            }
                                                }
                        
                    else:
                        # Merge new channel into existing dataframe
                        subset_data[filename]["data"] = pd.merge(
                                                                 subset_data[filename]["data"],
                                                                 subset_df.rename(columns={channel: channel}),
                                                                 on="Time (s)",
                                                                 how="outer"
                                                                )
     
                        # Add metadata for this channel
                        subset_data[filename]["metadata"][channel] = {
                                                                      "coral_id": coral_id,
                                                                      "volume": volume,
                                                                      "treatment": treatment,
                                                                     }
                       
            # Sort nested dataframe inside each filename dict
            # Unlikely to be needed because time is already increasing.
            for fname in subset_data:
                subset_data[fname]["data"] = (
                                              subset_data[fname]["data"]
                                              .sort_values("Time (s)")
                                              .reset_index(drop=True)
                                              )
                
        return subset_data if subset_data else None                      
    
    def slider_bars(self, raw_data_dropped_channels, file_name, tab):

        """
        Adds independent sliders for each channel in the dataframe
        and returns a new dataframe where each channel is masked to
        its own time window.
    
        Optionally allows the user to exclude a spike from the middle
        of the selected time window.
    
        Parameters
        ----------
        raw_data_dropped_channels : pd.DataFrame of the raw data selected by the user
            from the aggrid table of all files. Comes from subset_data.
            Must contain a 'Time (s)' column plus one or more channel columns.
    
        file_name : str
            Unique identifier for Streamlit widget keys.
    
        tab : str
            Whether the call came from tab1, tab2, or tab3.
            Needed for the widget keys.
    
        Returns
        -------
        edited_df : pd.DataFrame
            Copy of the dataframe with excluded values replaced by NaN.
    
        crop_info : dict
            Stores cropping information for each channel so plots can
            display excluded spike regions.
        """
    
        st.write("")
    
        st.markdown(
                    f"<p style='color: DarkBlue; \
                              font-size: 18px; \
                              margin: 0; \
                              font-style: italic;'>Select a time window for each channel.</p>",
                    unsafe_allow_html=True,
                   )
    
        st.write("")
    
        # Copy to avoid modifying original
        edited_df = raw_data_dropped_channels.copy()
    
        # Get the time data
        time = edited_df["Time (s)"]
    
        # Get existing channel cols since some may be deleted
        channel_cols = [c for c in edited_df.columns if c != "Time (s)"]
    
        # Store crop information for plotting later
        crop_info = {}
    
        # Loop through channels two at a time to build side-by-side (2 col) slider bars
        for i in range(0, len(channel_cols), 2):
    
            # Create three columns, one for a little space:
            # left slider, spacer, right slider
            col1, spacer, col2 = st.columns([1, 0.1, 1])
    
            # Get index and col name from channel cols in groups of 2
            for j, col in enumerate(channel_cols[i:i + 2]):
    
                target_col = col1 if j == 0 else col2
    
                with target_col:
    
                    # Print the channel number
                    st.write(f"**{col}**")
                 
                    # Outer crop slider (existing functionality)
                    # Crops from the ends
                    t_min, t_max = st.slider(
                                            f"Select time window for {col}",
                                            min_value=float(time.min()),
                                            max_value=float(time.max()),
                                            value=(float(time.min()), float(time.max())),
                                            step=1.0,
                                            label_visibility="collapsed",
                                            key=f"time_slider_{file_name}_{col}_{tab}",
                                           )
    
                    # Create a mask that includes the times
                    # the user wants to keep
                    outer_mask = (time >= t_min) & (time <= t_max)
    
                    # Optional spike exclusion
                    remove_spike = st.checkbox(
                                               "Exclude spike",
                                               value=False,
                                               key=f"spike_checkbox_{file_name}_{col}_{tab}",
                                              )
    
                    spike_start = None
                    spike_stop = None
    
                    if remove_spike:
    
                        # Default slider to the middle 10% of the
                        # selected time window
                        default_start = t_min + 0.45 * (t_max - t_min)
                        default_stop = t_min + 0.55 * (t_max - t_min)
    
                        spike_start, spike_stop = st.slider(
                                                            "Spike region",
                                                            min_value=float(t_min),
                                                            max_value=float(t_max),
                                                            value=(float(default_start),
                                                                   float(default_stop)),
                                                            step=1.0,
                                                            key=f"spike_slider_{file_name}_{col}_{tab}",
                                                          )
    
                        # Mask points inside the spike
                        spike_mask = (time >= spike_start) & \
                                     (time <= spike_stop)
    
                    else:
    
                        spike_mask = False

                    # Combine both masks
                    final_mask = outer_mask & (~spike_mask)
    
                    # Set all excluded values to NaN
                    edited_df.loc[~final_mask, col] = np.nan
    
                    # Save crop information for plotting
                    crop_info[col] = {
                                      "outer_start": t_min,
                                      "outer_stop": t_max,
                                      "spike_removed": remove_spike,
                                      "spike_start": spike_start,
                                      "spike_stop": spike_stop,
                                     }
    
        return edited_df, crop_info
    
    def plot_channels(self, edited_df, file_name, all_plots, crop_info=None):
        
        """
        Plots raw oxygen/signal data for each channel in one uploaded file.
    
        Parameters
        ----------
        edited_df : pandas.DataFrame
            Dataframe containing one time column, "Time (s)", and one or more
            channel columns.
    
        file_name : str
            Name of the file being plotted. Used in the plot title and dictionary key.
    
        all_plots : dict
            Dictionary used to store matplotlib figure objects for later export,
            such as PDF generation.
    
        crop_info : dict or None
            Optional dictionary containing spike/crop information for each channel.
            If a spike was removed, vertical dashed lines are drawn at the crop
            boundaries.
    
        Returns
        -------
        x_y_valid_dict : dict
            Dictionary containing valid non-NaN x/y arrays for each channel.
    
            Example:
            {
                "Ch1": [x_valid, y_valid],
                "Ch2": [x_valid, y_valid]
            }
    
        all_plots : dict
            Updated plot dictionary containing the raw figure for this file.
        """

        # Extract the time axis as a NumPy array.
        # This will be used as x-values for every channel.
        time = edited_df["Time (s)"].values
        
        # This dictionary will store the cleaned x/y data for each channel.
        # NaN y-values are removed before storing.
        x_y_valid_dict = {}
    
        plt.close("all")
        
        figsize = (4, 3)
    
        st.write("")
    
        # Create a Streamlit container for the plot.
        with st.container():
    
            fig_raw, ax = plt.subplots(figsize=figsize)
    
            # Loop through every channel signal column.
            for ch in edited_df.drop(columns=["Time (s)"]).columns:
    
                y = edited_df[ch].values
    
                # Create a Boolean mask identifying valid, non-NaN signal values.
                # Keep only valid data
                mask = ~np.isnan(y)
                x_valid = time[mask]
                y_valid = y[mask]
    
                # Store the cleaned x/y arrays for later regression or processing.
                x_y_valid_dict[ch] = [x_valid, y_valid]
    
                # Plot data
                ax.plot(x_valid, y_valid, label=ch)
    
                # If spike extracted
                if crop_info is not None:
                    
                    # Get crop information for this specific channel.
                    # If the channel is not in crop_info, use an empty dictionary.
                    info = crop_info.get(ch, {})
    
                    # Draw crop markers if spike was removed
                    if info.get("spike_removed"):
                        spike_start = info.get("spike_start")
                        spike_stop = info.get("spike_stop")
    
                        if spike_start is not None and spike_stop is not None:
                            ax.axvline(spike_start, linestyle="--", linewidth=1)
                            ax.axvline(spike_stop, linestyle="--", linewidth=1)
    
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"O$_2$ ($\mu$mol)")
            ax.set_title(f"Raw: {file_name}")
    
            ax.legend(
                      loc="center left",
                      bbox_to_anchor=(1, 0.5),
                      frameon=False
                     )
    
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig_raw, clear_figure=False)
    
            all_plots[f"{file_name}_raw"] = fig_raw
    
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
    
        # Convert time axis from seconds to hours
        x_hr = np.array(x) / 3600
    
        # Regress
        slope, intercept, r, p, stderr = linregress(x_hr, y)
        
        # Get the number of points in the dataset
        n = len(x_hr)
        
        # Calculate the degrees of freedom. Make sure this is at least 0.
        df_resid = max(0, n - 2)

        # Initialize confidence intervals
        ci_low = np.nan
        ci_high = np.nan
        slope_ci = "NA"

        if df_resid > 0:
            # Use scipy t.ppf to calculate the student's t score 
            t_val = t.ppf(1 - 0.025, df_resid)
            # Calculate low and high CI for 95% CI
            ci_low = slope - t_val * stderr
            ci_high = slope + t_val * stderr
            slope_ci = f"{ci_low:.4g} – {ci_high:.4g}"
            
        # Calculate R2
        R2 = r*r
        
        # Calculate sum of squared residuals
        # Predict y values from regression
        y_pred = intercept + slope * x_hr
        # Calculate residuals
        residuals = np.array(y) - y_pred
        # Calculate sum of squared residuals
        ssr = np.sum(residuals**2)
        
        regress_results = {
                            "slope (umol/hr)":slope,
                            "intercept":intercept,
                            "R2":R2,
                            "slope pval":p,
                            "slope stderr (umol/hr)":stderr,
                            "CI Low (umol/hr)":ci_low,
                            "CI High (umol/hr)":ci_high,
                            "slope 95% CI (umol/hr)":slope_ci,
                            "squared residuals":ssr
                           }
        
        return regress_results, residuals
    
    #-------------------------------------------------------------------------------------------------
    # Blank subtraction plus helper functions
    def blank_subtraction(
                          self,
                          all_subset_regs,
                          blank_id="Blank",
                          slope_key="slope (umol/hr)",
                          se_key="slope stderr (umol/hr)",
                          corrected_se_key="Corrected slope stderr (umol/hr)",
                          coral_id_key="Coral ID",
                          ci_multiplier=1.96,
                         ):
        
        """
        Applies blank subtraction to regression results.
        For each sample slope, the method subtracts the slope from the matching blank:
    
            blank-corrected slope = sample slope - blank slope
    
        It also propagates uncertainty from both the sample and blank:
    
            blank-corrected SE = sqrt(sample_SE^2 + blank_SE^2)
    
        The same logic is applied to both the regular slope SE and the
        lag-adjusted/Newey-West corrected SE.
    
        Blanks are matched either:
        1. directly from the same filename, if that file contains a blank channel, or
        2. from another file in the same date/run/light-dark group.
        """
    
        blank_by_filename = {}
    
        # Find files containing blanks
        for filename, channel_dict in all_subset_regs.items():
            for ch, reg_data in channel_dict.items():
    
                if self.is_blank_row(reg_data, coral_id_key, blank_id):
                    blank_by_filename[filename] = {
                                                   "channel": ch,
                                                   "data": reg_data,
                                                  }
                    break
    
        # Store a blank by experimental run
        blank_by_group = {}
    
        # Group blanks by same date/run/light-dark condition
        for filename, blank_info in blank_by_filename.items():
    
            # Extract the grouping information from the filename.
            # Example group might be something like:
            # ("2026-06-01", "Run 2", "Dark")
            group = self.get_blank_group(filename)
    
            # If this group does not already have a blank assigned,
            # use this blank as the representative blank for the group.
            if group not in blank_by_group:
                blank_by_group[group] = {
                                         "filename": filename,
                                         "channel": blank_info["channel"],
                                         "data": blank_info["data"],
                                        }
    
        # Apply blank correction
        for filename, channel_dict in all_subset_regs.items():
    
            # Determine which group this file belongs to.
            group = self.get_blank_group(filename)
    
            # Case 1:
            # This file contains its own blank channel.
            if filename in blank_by_filename:
                blank_filename = filename
                blank_data = blank_by_filename[filename]["data"]
    
            # Case 2:
            # This file does not contain a blank, but another file from the
            # same date/run/light-dark group does.
            elif group in blank_by_group:
                blank_filename = blank_by_group[group]["filename"]
                blank_data = blank_by_group[group]["data"]
    
            # Case 3:
            # No matching blank was found.
            else:
                blank_filename = "Blank File"
                blank_data = None
    
            # If no blank exists for this file/group, mark the result
            # and clear any previous blank-corrected values.
            if blank_data is None:
                for reg_data in channel_dict.values():
                    reg_data["Blank Subtracted"] = "Blank File"
                    self.clear_blank_results(reg_data)
    
                continue
    
            # Pull the blank slope and uncertainty values.
            blank_slope = blank_data.get(slope_key)
            blank_se = blank_data.get(se_key)
            blank_corr_se = blank_data.get(corrected_se_key)
            
            # Apply the selected blank to every channel in this file
            for reg_data in channel_dict.values():
    
                reg_data["Blank Subtracted"] = blank_filename
    
                # If this row is itself the blank, define its blank-corrected
                # slope as zero because blank - blank = 0.
                if self.is_blank_row(reg_data, coral_id_key, blank_id):
                    
                    reg_data["blank-corrected slope (umol/hr)"] = 0.0
                    reg_data["blank-corrected slope SE (umol/hr)"] = None
                    reg_data["blank-corrected 95% CI (umol/hr)"] = None
                    reg_data["blank-corrected Lag-adjusted SE (umol/hr)"] = None
                    reg_data["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = None
                    
                    continue
    
                # Get the sample slope.
                sample_slope = reg_data.get(slope_key)
    
                if sample_slope is None or blank_slope is None:
                    
                    self.clear_blank_results(reg_data)
                    
                    continue
    
                # Calculate blank-corrected slope
                bc_slope = sample_slope - blank_slope
                reg_data["blank-corrected slope (umol/hr)"] = bc_slope
    
                # Propogate standard error and calculate new 95% CI
                sample_se = reg_data.get(se_key)
    
                if sample_se is not None and blank_se is not None:
                    
                    bc_se = math.sqrt(sample_se**2 + blank_se**2)
                    reg_data["blank-corrected slope SE (umol/hr)"] = bc_se
                    reg_data["blank-corrected CI Low (umol/hr)"] = bc_slope - ci_multiplier * bc_se
                    reg_data["blank-corrected CI High (umol/hr)"] = bc_slope + ci_multiplier * bc_se
                    reg_data["blank-corrected 95% CI (umol/hr)"] = (
                                                                      f"{reg_data['blank-corrected CI Low (umol/hr)']:.3f} – "
                                                                      f"{reg_data['blank-corrected CI High (umol/hr)']:.3f}"
                                                                     )
                    
                    
                    
                else:
                    reg_data["blank-corrected slope SE (umol/hr)"] = None
                    reg_data["blank-corrected CI Low (umol/hr)"] = None
                    reg_data["blank-corrected CI High (umol/hr)"] = None
                    reg_data["blank-corrected 95% CI (umol/hr)"] = None
    
                # Do same error propogation for Newey-West correction
                sample_corr_se = reg_data.get(corrected_se_key)
    
                if sample_corr_se is not None and blank_corr_se is not None:
                    bc_corr_se = math.sqrt(sample_corr_se**2 + blank_corr_se**2)
    
                    reg_data["blank-corrected Lag-adjusted SE (umol/hr)"] = bc_corr_se
                    reg_data["blank-corrected Lag-adjusted CI Low (umol/hr)"] = (
                                                                                   bc_slope - ci_multiplier * bc_corr_se
                                                                                  )
                    reg_data["blank-corrected Lag-adjusted CI High (umol/hr)"] = (
                                                                                    bc_slope + ci_multiplier * bc_corr_se
                                                                                   )
                    reg_data["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = (
                                                                                   f"{reg_data['blank-corrected Lag-adjusted CI Low (umol/hr)']:.3f} – "
                                                                                   f"{reg_data['blank-corrected Lag-adjusted CI High (umol/hr)']:.3f}"
                                                                                  )
                
                else:
                    reg_data["blank-corrected Lag-adjusted SE (umol/hr)"] = None
                    reg_data["blank-corrected Lag-adjusted CI Low (umol/hr)"] = None
                    reg_data["blank-corrected Lag-adjusted CI High (umol/hr)"] = None
                    reg_data["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = None
                    
                
                    
    
        return all_subset_regs
    
    def is_blank_row(self, reg_data, coral_id_key="Coral ID", blank_id="Blank"):
        """
        Determine if an ID is a blank.
        """
        coral_id = str(reg_data.get(coral_id_key, "")).strip().lower()
        return coral_id == blank_id.lower()
    
    def get_blank_group(self, filename):
        """
        Removes letter from Run # in the filename.
        """
    
        fname = str(filename)
    
        # 08_01_24_Run1A_Dark  -> 08_01_24_Run1_Dark
        # 08_01_24_Run1B_Light -> 08_01_24_Run1_Light
        fname = re.sub(r"(Run\d+)[A-Za-z](_(?:Dark|Light))$", r"\1\2", fname)
    
        return fname
    
    def clear_blank_results(self, reg_data):
        reg_data["blank-corrected slope (umol/hr)"] = None
        reg_data["blank-corrected slope SE (umol/hr)"] = None
        reg_data["blank-corrected CI Low (umol/hr)"] = None
        reg_data["blank-corrected CI High (umol/hr)"] = None
        reg_data["blank-corrected 95% CI (umol/hr)"] = None
        reg_data["blank-corrected Lag-adjusted SE (umol/hr)"] = None
        reg_data["blank-corrected Lag-adjusted CI Low (umol/hr)"] = None
        reg_data["blank-corrected Lag-adjusted CI High (umol/hr)"] = None
        reg_data["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = None
        
    #-------------------------------------------------------------------------------------------------

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
    
    def newey_west_analysis(self, x, y, lags, alpha=0.05):
        """
        Apply Newey-West (HAC) correction to a single regression using
        a precomputed lag value.
    
        Parameters
        ----------
        x : array-like
            1D x values for one channel.
        y : array-like
            1D y values for one channel.
        lags : int
            Lag value already computed from the residuals.
        alpha : float
            Significance level for CI (default 0.05 -> 95% CI).
    
        Returns
        -------
        dict
            Dictionary containing HAC-corrected slope standard error,
            p-value, and confidence interval.
        """
    
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Convert time axis from seconds to hours
        x = x / 3600
    
        # Two-tailed z-score for confidence interval
        z = norm.ppf(1 - alpha / 2)
    
        # Build regression matrix
        X = sm.add_constant(x)
    
        # Ordinary OLS fit
        ols_model = sm.OLS(y, X).fit()
    
        # Newey-West / HAC corrected fit using the supplied lag value
        nw_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    
        # Slope is the same as OLS; only inference changes
        slope = float(ols_model.params[1])
        se_slope = float(nw_model.bse[1])
        pval_slope = "{:.3e}".format(float(nw_model.pvalues[1]))
    
        ci_low = slope - z * se_slope
        ci_high = slope + z * se_slope
        nw_slope_ci = f"{ci_low:.4g} – {ci_high:.4g}"
    
        return {
                "Corrected slope stderr (umol/hr)": se_slope,
                # "NW slope pval": pval_slope,
                "Corrected slope 95% CI (umol/hr)": nw_slope_ci,
               }
    
    def ESS_correction_from_residuals(self, residuals, max_lag=200):
        """
        Compute effective sample size (ESS) from a 1D residual array.
    
        Parameters
        ----------
        residuals : array-like
            Residuals from a single regression fit.
        max_lag : int
            Maximum lag to consider when estimating ESS.
    
        Returns
        -------
        ess : float
            Effective sample size.
        """
    
        residuals = np.asarray(residuals, dtype=float)
        N = len(residuals)
    
        if N < 2:
            return float(N)
    
        # Determine max lag
        lags = self.estimate_maxlags(residuals, max_lag=max_lag)
        lags = min(lags, N - 1)
    
        if lags < 1:
            return float(N)
    
        # Compute autocorrelations up to lag, skipping lag 0
        acf_vals = acf(residuals, nlags=lags, fft=True)[1:]
    
        # Compute ESS
        denom = 1 + 2 * np.sum(
                                ((N - np.arange(1, len(acf_vals) + 1)) / N) * acf_vals
                              )
    
        # Guard against pathological values
        if denom <= 0:
            return float(N)
    
        ess = N / denom
    
        # ESS should not exceed the actual number of points
        ess = min(ess, N)
    
        return float(ess)
        
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
    
    def plot_residuals(self, residuals, ax, max_points=2000):
        """
        Generates a plot of residuals. 
        Downsizes large datasets for speed.
        """
        
        # Get residuals as np array
        r = np.asarray(residuals)
        # Get the size of the array
        n = r.size
    
        # Max_points downselects the data so make sure there are enough points
        if n > max_points:
            # Select evenly spaced points across the dataset
            idx = np.linspace(0, n - 1, max_points).astype(int)
            r = r[idx]
            x = idx
        
        else:
            # Or if dataset is small just keep all the points
            x = np.arange(n)
    
        ax.plot(x, r, marker=".", linestyle="None", markersize=2, alpha=0.6, rasterized=True)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Index")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals around 0")
    
    def plot_residual_acf(self, residuals, ax, max_lag=200, hard_cap=50):
        """
        Creates a plot of residual autocorrelation. 
        Caps the plot at 50 points
        from the first point for speed.
        """
        
        # Create np array
        r = np.asarray(residuals)
    
        # Calculates a reasonable number of lags to consider based on the data
        best_lag = int(self.estimate_maxlags(r, max_lag=max_lag))
        # Enforces limits on the lags to be calculated-no more than hard cap
        # but if number of points is less hard cap use those
        best_lag = max(1, min(best_lag, hard_cap, r.size - 1))
    
        # Calculates acf
        acf_vals = acf(r, nlags=best_lag, fft=True)
        lags = np.arange(1, best_lag + 1)
    
        ax.vlines(lags, 0, acf_vals[1:], linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlim(0, best_lag + 1)
        ax.set_title(f"ACF of Residuals (maxlag={best_lag})")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
    
    def plot_residual_distribution(self, residuals, ax, bins=20):
        """Plots a histogram of residuals."""
        
        r = np.asarray(residuals)
    
        # Fast "fit"
        mu = r.mean()
        std = r.std(ddof=1) if r.size > 1 else 0.0
    
        # Fast histogram
        hist, edges = np.histogram(r, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, hist, width=(edges[1]-edges[0]), alpha=0.6, edgecolor="none")
    
        if std > 0:
            x = np.linspace(edges[0], edges[-1], 200)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, linestyle="--", linewidth=2)
    
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
    
    def build_pdf(self, all_subset_residuals, max_lag=200, hard_cap=50, dpi=120) -> bytes:
        
        """
        Calls the plot functions above (residuals, acf, residual distribution)
        and builds a pdf file containing all plots.
        """
        
        total_plots = sum(len(channels) for channels in all_subset_residuals.values())
        
        if total_plots == 0:
            return b""
    
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for filename, channels in all_subset_residuals.items():
                for channel, residuals in channels.items():
                    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=False)
                   
                    fig.suptitle(f"{filename} — {channel}", fontsize=12)
                    
                    # 0.92 adds space betwee title and plot
                    fig.tight_layout(rect=[0, 0, 1, 0.92]) 
                    
                    self.plot_residuals(residuals, axes[0])
                    self.plot_residual_acf(residuals, axes[1], max_lag=max_lag, hard_cap=hard_cap)
                    self.plot_residual_distribution(residuals, axes[2])
    
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)  
    
        return buf.getvalue()
    
    def render_plots(self, all_subset_residuals, max_lag=200, hard_cap=50):
        
        """
        Renders the diagnostic plots.
        """
        
        # Make sure there data to plot
        total_plots = sum(len(channels) for channels in all_subset_residuals.values())
        
        if total_plots == 0:
            st.warning("No plots to render.")
            return
    
        # Start a progress bar
        bar = st.progress(0.0)
        txt = st.empty()
        # Count files rendered
        i = 0
    
        # Generate 3 side-by-side plots for each channel inside there own st.expander()
        # Expander speeds things up and makes it more readable
        for filename, channels in all_subset_residuals.items():
            for channel, residuals in channels.items():
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
                # fig.suptitle(f"{filename} — {channel}", fontsize=12)
    
                self.plot_residuals(residuals, axes[0])
                self.plot_residual_acf(residuals, axes[1], max_lag=max_lag, hard_cap=hard_cap)
                self.plot_residual_distribution(residuals, axes[2])
    
                with st.expander(f"{filename} — {channel}", expanded=False):
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
    
                plt.close(fig) 
                
                i += 1
                bar.progress(i / total_plots)
                txt.text(f"Rendered {i}/{total_plots}")
    
    def rolling_regression_xy(self, x, y, window_seconds, step_perc=20.0):
        """
        Calculate rolling OLS regressions for one x/y trace.
    
        Rolling windows are defined by elapsed time rather than number of
        observations so that the same window length is used even when sampling
        frequency differs among runs.
    
        Parameters
        ----------
        x, y : array-like
            Time in seconds and O2 amount in umol.
    
        window_seconds : float
            Length of each rolling regression window in seconds.
    
        step_perc : float
            Window advance as a percentage of the window length.
            For example, 20% with a 600-second window advances each
            successive window by 120 seconds.
        """
    
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    
        # Remove non-finite x/y pairs.
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
    
        if len(x) < 5:
            return pd.DataFrame()
    
        # Make sure the data are in chronological order.
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    
        x_min = float(np.min(x))
        x_max = float(np.max(x))
    
        window_seconds = float(window_seconds)
    
        # Trace must be long enough to contain at least one complete window.
        if (x_max - x_min) < window_seconds:
            return pd.DataFrame()
    
        # A 20% step means a 600-second window advances by 120 seconds.
        step_seconds = window_seconds * float(step_perc) / 100.0
    
        if step_seconds <= 0:
            return pd.DataFrame()
    
        final_start = x_max - window_seconds
    
        starts = list(np.arange(x_min, final_start + 1e-9, step_seconds))
    
        # Include the final possible full window so the end of the run is
        # represented even when it does not fall exactly on the regular grid.
        if final_start >= x_min:
            if not starts or abs(starts[-1] - final_start) > 1e-6:
                starts.append(float(final_start))
    
        results = []
    
        for window_number, start_time in enumerate(starts, start=1):
    
            end_time = start_time + window_seconds
            mask = (x >= start_time) & (x <= end_time)
    
            # Require enough observations for a meaningful regression.
            if np.count_nonzero(mask) < 5:
                continue
    
            xi = x[mask]
            yi = y[mask]
    
            # Convert seconds to hours so slope is reported in umol/hr.
            xi_hr = xi / 3600.0
    
            X = sm.add_constant(xi_hr)
            model = sm.OLS(yi, X).fit()
    
            slope = float(model.params[1])
            slope_se = float(model.bse[1])
    
            if abs(slope) > 1e-12:
                slope_rse = float(100.0 * slope_se / abs(slope))
                
            else:
                slope_rse = np.nan
    
            ci = model.conf_int(alpha=0.05)
            slope_ci_low = float(ci[1, 0])
            slope_ci_high = float(ci[1, 1])
    
            results.append({
                            "Window": window_number,
                            "Start": float(xi[0]),
                            "End": float(xi[-1]),
                            "Raw slope (umol/hr)": slope,
                            "R2": float(model.rsquared),
                            "Raw slope SE (umol/hr)": slope_se,
                            "Raw slope %RSE": slope_rse,
                            "Raw slope 95% CI (umol/hr)": (f"{slope_ci_low:.3f} - {slope_ci_high:.3f}"),
                            "intercept": float(model.params[0]),
                            "SSR": float(np.sum(model.resid ** 2)),
                         })
    
        return pd.DataFrame(results)
    
    def rate_shift_analysis(self,
                            x,
                            y,
                            window_seconds=None,
                            overlap_perc=None):
        """
        Scan a trace for a substantial, sustained change between adjacent
        long-window respiration-rate estimates.
    
        The diagnostic uses 5-minute regressions by default. Consecutive
        windows overlap by only 20% (an 80% advance), and the final possible
        5-minute window is always included so the end of the run is sampled.
    
        The Rate Shift diagnostic is not applied to traces whose full-run
        absolute slope is below RATE_SHIFT_MIN_SLOPE. These traces are
        considered essentially flat overall, so local slope differences are
        not interpreted as meaningful physiological rate shifts. Rate Shift
        is reported as N/A.
    
        For eligible traces, each pair of adjacent windows is considered only
        when at least one of the two slopes reaches RATE_SHIFT_MIN_SLOPE. Pairs
        in which both slopes are below this threshold are ignored. For each
        eligible pair, the relative slope difference is calculated as:
    
            |slope_2 - slope_1| / max(|slope_1|, |slope_2|) * 100
    
        A qualifying change must exceed RATE_SHIFT_PERCENT_THRESHOLD and must
        persist into the following 5-minute window. The third slope must remain
        beyond the midpoint between the first and second slopes in the
        direction of the proposed shift.
    
        This is a diagnostic flag, not an automatic rejection criterion. It is
        intended to tell the analyst that the estimated respiration rate
        changed substantially during the selected interval and the plot should
        be inspected.
        """
        
        # Convert to np arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Clean
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        
        # Define variables
        if window_seconds is None:
            window_seconds = self.RATE_SHIFT_WINDOW_SECONDS
    
        if overlap_perc is None:
            overlap_perc = self.RATE_SHIFT_OVERLAP_PERC
    
        window_seconds = float(window_seconds)
        overlap_perc = float(overlap_perc)
    
        # Create a dict for results
        empty_result = {
                        "Rate Shift": "",
                        "Rate Shift Eligible": False,
                        "Rate Shift %": np.nan,
                        "Rate Shift Absolute (umol/hr)": np.nan,
                        "Rate Shift Method": "",
                        "Rate Shift First Slope (umol/hr)": np.nan,
                        "Rate Shift Second Slope (umol/hr)": np.nan,
                        "Rate Shift Third Slope (umol/hr)": np.nan,
                        "Rate Shift First Window": "",
                        "Rate Shift Second Window": "",
                        "Rate Shift Third Window": "",
                        "Rate Shift Windows": 0,
                      }
    
        # Make sure there is enough data
        if x.size < 3 or window_seconds <= 0:
            return empty_result
    
        # Get min, max, and duration of the experiment
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        duration = x_max - x_min
    
        if duration < window_seconds:
            return empty_result
    
        # Define thresholds
        min_slope = float(self.RATE_SHIFT_MIN_SLOPE)
        percent_threshold = float(self.RATE_SHIFT_PERCENT_THRESHOLD)
    
        # Determine whether the full trace is eligible for
        # Rate Shift analysis.
        # If the full-run slope is essentially flat, local slope
        # fluctuations are not interpreted as physiological rate shifts.
        
        # Convert to hours
        x_hr = x / 3600.0
        # Set to non-zero y-intercept
        X_full = sm.add_constant(x_hr)
        full_model = sm.OLS(y, X_full).fit()
        full_slope = float(full_model.params[1])
    
        # Decide whether the run is eligible for slope change analysis (overall slope > 1)
        eligible = bool(abs(full_slope) >= min_slope)
    
        if not eligible:
            result = empty_result.copy()
            result["Rate Shift"] = "Low Overall Rate"
            result["Rate Shift Eligible"] = False
            return result
    
        # Now check for slope changes during the run
        # Build the sequence of long windows across the run.
        # Calculate how far to step from one sliding window to the next
        step_seconds = window_seconds * (1.0 - overlap_perc / 100.0)
        # Don't let the step ever be 0 seconds-like if we entered 100% overlap
        step_seconds = max(step_seconds, 1e-9)
        
        # Calculate the latest possible window start time
        final_start = x_max - window_seconds
        
        # List for window start times
        starts = []
        current = x_min
        # This deals with floating point logic
        tol = 1e-9
    
        # Generates the window start times
        while current <= final_start + tol:
            starts.append(float(current))
            current += step_seconds
    
        # Always include the final possible window so the end of the run
        # is represented even when it does not fall on the regular grid.
        # The last 5 minutes is always a window even if it overlaps with the previous 
        # window by more than 20%
        if final_start >= x_min:
            if not starts or abs(starts[-1] - final_start) > 1e-6:
                starts.append(float(final_start))
    
        # Create a list for the regressions of each window
        windows = []
    
        # Loop through windows
        for start_time in starts:
            end_time = start_time + window_seconds
            mask = (x >= start_time) & (x <= end_time)
    
            # Normal FireSting windows contain hundreds of observations.
            if np.count_nonzero(mask) < 5:
                continue
            
            # Get xy data for the window
            xi = x[mask]
            yi = y[mask]
    
            # Convert to hours
            xi_hr = xi / 3600.0
    
            # Include a non-zero regression intercept.
            X = sm.add_constant(xi_hr)
            model = sm.OLS(yi, X).fit()
            
            # Append results to list as a dict
            windows.append({
                            "Start": float(xi[0]),
                            "End": float(xi[-1]),
                            "Slope": float(model.params[1]),
                            "SE": float(model.bse[1]),
                            "R2": float(model.rsquared),
                          })
    
        if len(windows) < 3:
            result = empty_result.copy()
            result["Rate Shift Windows"] = len(windows)
            return result
    
        # Start a list for window slope comparisons
        comparisons = []
    
        # Compare adjacent long-window slopes.
        # Ignore pairs in which both slopes are below the minimum
        # meaningful rate. A large relative change is counted as a
        # Rate Shift only when it persists into the following window.
        
        # Loop through windows
        for i in range(len(windows) - 2):
    
            first = windows[i]
            second = windows[i + 1]
            third = windows[i + 2]
    
            slope_1 = float(first["Slope"])
            slope_2 = float(second["Slope"])
            slope_3 = float(third["Slope"])
    
            # Do not analyze changes between two essentially flat windows.
            if max(abs(slope_1), abs(slope_2)) < min_slope:
                continue
    
            abs_diff = abs(slope_2 - slope_1)
    
            denom = max(abs(slope_1), abs(slope_2))
    
            percent_diff = 100.0 * abs_diff / denom
    
            threshold_pass = (percent_diff >= percent_threshold)
    
            score = (percent_diff / max(percent_threshold, 1e-12))
    
            # Persistence:
            # after a qualifying change, the next slope must remain
            # on the new side of the midpoint between the first and
            # second slopes.
            midpoint = 0.5 * (slope_1 + slope_2)
    
            if slope_2 > slope_1:
                persists = slope_3 > midpoint
    
            elif slope_2 < slope_1:
                persists = slope_3 < midpoint
    
            else:
                persists = False
    
            flag = bool(threshold_pass and persists)
    
            comparisons.append({
                                "index": i,
                                "flag": flag,
                                "score": float(score),
                                "method": "Relative",
                                "abs_diff": float(abs_diff),
                                "percent_diff": float(percent_diff),
                                "slope_1": slope_1,
                                "slope_2": slope_2,
                                "slope_3": slope_3,
                              })
    
        # The trace had a meaningful full-run slope, but there may be
        # no adjacent window pairs large enough to evaluate.
        if not comparisons:
            result = empty_result.copy()
            result["Rate Shift"] = "No"
            result["Rate Shift Eligible"] = True
            result["Rate Shift Windows"] = len(windows)
            return result
    
        # If any comparison is flagged, report the strongest flagged
        # comparison. Otherwise report the comparison that came closest
        # to the threshold for diagnostic purposes.
        flagged = [c for c in comparisons if c["flag"]]
    
        candidates = (flagged if flagged else comparisons)
    
        best = max(candidates, key=lambda c: c["score"])
    
        i = best["index"]
    
        return {
                "Rate Shift": (
                                "Yes"
                                if bool(flagged)
                                else "No"
                              ),
                "Rate Shift Eligible": True,
                "Rate Shift %": float(best["percent_diff"]),
                "Rate Shift Absolute (umol/hr)": float(best["abs_diff"]),
                "Rate Shift Method": best["method"],
                "Rate Shift First Slope (umol/hr)": float(best["slope_1"]),
                "Rate Shift Second Slope (umol/hr)": float(best["slope_2"]),
                "Rate Shift Third Slope (umol/hr)": float(best["slope_3"]),
                "Rate Shift First Window": (
                                            f'{windows[i]["Start"]:.1f} - '
                                            f'{windows[i]["End"]:.1f}'
                                           ),
                "Rate Shift Second Window": (
                                             f'{windows[i + 1]["Start"]:.1f} - '
                                             f'{windows[i + 1]["End"]:.1f}'
                                            ),
                "Rate Shift Third Window": (
                                            f'{windows[i + 2]["Start"]:.1f} - '
                                            f'{windows[i + 2]["End"]:.1f}'
                                           ),
                                           "Rate Shift Windows": int(len(windows)),
               }

    @st.cache_data(max_entries=20, ttl=3600)
    def compute_rolling_regression(_self, 
                                   final_x_y_valid_dict, 
                                   all_subset_regs, 
                                   window_seconds, 
                                   step_perc=20):
     
        """
        Calculate rolling regressions for the Rolling Regression tab.
        
        The numerical fits come from rolling_regression_xy(), which is also
        used by the Linear Regression rate-stability diagnostic. All rolling
        windows are returned here; rolling_reg_ui() shows 10 candidate windows
        ranked by closeness to the most common KDE rate so the existing
        click-to-inspect workflow is preserved without relying on R2 ranking.
        """
        results = []
        progress = st.progress(0)
        total_files = len(final_x_y_valid_dict)
        
        for file_idx, (filename, channels) in enumerate(final_x_y_valid_dict.items(), start=1):
            for channel, xy in channels.items():
                x, y = np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)
                
                coral_id = np.nan
                if filename in all_subset_regs and channel in all_subset_regs[filename]:
                    coral_id = all_subset_regs[filename][channel].get("Coral ID", np.nan)
                
                rolling = _self.rolling_regression_xy(x, y, window_seconds, step_perc)
                
                if rolling.empty:
                    results.append({
                                    "Filename": filename,
                                    "Coral ID": coral_id,
                                    "Channel": channel,
                                    "Window": "Selected window width too large",
                                    "Start": np.nan,
                                    "End": np.nan,
                                    "Raw slope (umol/hr)": np.nan,
                                    "R2": np.nan,
                                    "Raw slope SE (umol/hr)": np.nan,
                                    "Raw slope %RSE": np.nan,
                                    "Raw slope 95% CI (umol/hr)": "Not enough rolling windows",
                                    "intercept": np.nan,
                                    "SSR": np.nan,
                                   })
                    continue
                
                rolling.insert(0, "Channel", channel)
                rolling.insert(0, "Coral ID", coral_id)
                rolling.insert(0, "Filename", filename)
                
                # Match the existing display precision while keeping all windows.
                rolling["Raw slope (umol/hr)"] = rolling["Raw slope (umol/hr)"].round(3)
                rolling["R2"] = rolling["R2"].round(2)
                rolling["Raw slope SE (umol/hr)"] = rolling["Raw slope SE (umol/hr)"].round(3)
                rolling["Raw slope %RSE"] = rolling["Raw slope %RSE"].round(2)
                
                rolling["Raw slope 95% CI (umol/hr)"] = (
                                                        rolling["Raw slope 95% CI (umol/hr)"].apply(
                                                            lambda ci: f"{ci[0]:.3f} - {ci[1]:.3f}"
                                                            if isinstance(ci, (list, tuple, np.ndarray)) and len(ci) == 2
                                                            else ci
                                                        )
                                                    )
                
                rolling["intercept"] = rolling["intercept"].round(3)
                rolling["SSR"] = rolling["SSR"].round(3)
                results.extend(rolling.to_dict(orient="records"))
            
            progress.progress(file_idx / total_files)
        
        return pd.DataFrame(results)
    
    
    def rolling_rate_metrics(self,
                             slopes,
                             grid_n: int = 400):
        """
        Summarize the distribution of rolling-regression slopes.
    
        Returns the KDE modal slope, relative slope variability, and a slope
        assessment based on the variability of the rolling slopes. The KDE mode
        is also used to rank candidate rolling windows in the Rolling Regression
        tab.
        """
    
        slopes = np.asarray(slopes, dtype=float)
        slopes = slopes[np.isfinite(slopes)]
    
        if slopes.size < self.ROLLING_MIN_WINDOWS:
            
            return {
                    "mode": np.nan,
                    "variability_percent": np.nan,
                    "assessment": "Not enough rolling windows",
                   }
    
        sd = float(np.std(slopes, ddof=0))
    
        # If all rolling slopes are essentially identical, KDE is unnecessary.
        if not np.isfinite(sd) or sd < 1e-12:
            mode = float(np.median(slopes))
    
            if abs(mode) < self.ROLLING_MIN_SLOPE:
                variability_percent = np.nan
                assessment = "—"
            else:
                variability_percent = 0.0
                assessment = "Stable slope"
    
            return {
                    "mode": mode,
                    "variability_percent": variability_percent,
                    "assessment": assessment,
                   }
    
        # Find the most common rolling slope from the dominant KDE peak.
        kde_obj = gaussian_kde(slopes)
        grid = np.linspace(
                           float(np.min(slopes)),
                           float(np.max(slopes)),
                           int(grid_n)
                          )
    
        dens = kde_obj(grid)
        mode_idx = int(np.argmax(dens))
        mode = float(grid[mode_idx])
    
        # Percentage variability is not meaningful when the modal slope is near
        # zero because the small denominator can produce very large percentages.
        if abs(mode) < self.ROLLING_MIN_SLOPE:
            variability_percent = np.nan
            assessment = "—"
    
        else:
            variability_percent = float(100.0 * sd / abs(mode))
    
            if variability_percent <= self.ROLLING_STABLE_VARIABILITY_PERCENT:
                assessment = "Stable slope"
    
            elif variability_percent <= self.ROLLING_HIGH_VARIABILITY_PERCENT:
                assessment = "Variable slope — inspect for stable region"
    
            else:
                assessment = "Highly variable slope — single rate may not be appropriate"
    
        return {
                "mode": mode,
                "variability_percent": variability_percent,
                "assessment": assessment,
               }

    def rolling_reg_ui(self, 
                       final_x_y_valid_dict, 
                       results_df,
                       top_n_display: int = 10,
                       base_row_height: int = 30, 
                       max_rows: int = 20,
                       grid_key = "aggrid_table"):
        
        if results_df.empty:
            st.warning("Not enough data for rolling regression.  \n"
                       "The x-axis width was set less than the sliding window width.")
            return
        
        # Show the rolling windows whose slopes are closest to the most common
        # KDE slope for each Filename x Channel. R2 remains visible as a
        # diagnostic but is no longer the ranking criterion.
        df_show = results_df.copy()

        df_show["Rate Difference"] = np.nan
        df_show["_Most Common Slope"] = np.nan
        for (fname, ch), g in df_show.groupby(["Filename", "Channel"], sort=False):
            slopes = pd.to_numeric(g["Raw slope (umol/hr)"], errors="coerce").to_numpy(dtype=float)
            metrics = self.rolling_rate_metrics(slopes)
            mode = metrics["mode"]
            if np.isfinite(mode):
                idx = g.index
                df_show.loc[idx, "_Most Common Slope"] = mode
                df_show.loc[idx, "Rate Difference"] = np.abs(
                    pd.to_numeric(g["Raw slope (umol/hr)"], errors="coerce") - mode
                )
    
        # Preserve filename order as it appears in the dataframe
        # Build a dict of filenames and their indeces and add a column to df_show
        # The indeces allow the filename order to be converted to an integer order
        fname_order = {f: i for i, f in enumerate(df_show["Filename"].dropna().unique())}
        # Create a column of the integers. Any NaN are assigned a huge number and get pushed
        # to the bottom of the stack
        df_show["_fname_order"] = df_show["Filename"].map(fname_order).fillna(10**9).astype(int)
        
        # Now sort the channel numbers so the resulting order will be filename, the Ch1, Ch2...
        # Extract channel number if present (Ch1 -> 1). Non-channels go last.
        ch_num = (
                  df_show["Channel"]
                  .astype(str)
                  .str.extract(r'(?i)ch\s*(\d+)')[0]   
                  .astype(float)                       
                 )
        
        # Replace NaN with large number so non-channels sort last
        df_show["_ch_order"] = ch_num.fillna(999).astype(int)
        
        # Select top N per Filename+Channel by closeness to the KDE modal slope.
        # R2 is used only as a tie-breaker so it remains useful without deciding
        # which windows are shown.
        df_show = (
                   df_show.sort_values(["Rate Difference", "R2"],
                                       ascending=[True, False])
                       .groupby(["Filename", "Channel"], sort=False, as_index=False)
                       .head(top_n_display)
                  )
        
        # Final display order:
        # Filename (original order) -> channel (Ch1..Ch4) -> closest-to-common-slope first
        df_show = (
                   df_show.sort_values(["_fname_order", "_ch_order", "Rate Difference"],
                                    ascending=[True, True, True])
                       .drop(columns=["_fname_order", "_ch_order"])
                       .reset_index(drop=True)
                  )
    
        # Use consistent user-facing slope terminology in the rolling-regression
        # table. The underlying calculation fields retain their original names
        # until this point so the analysis code is unchanged. ASCII "umol/hr"
        # is used in AgGrid headings because the micro symbol does not render
        # reliably in all AgGrid/browser combinations.
        df_show = df_show.rename(columns={
                                         "Raw slope (umol/hr)": "Slope (umol/hr)",
                                         "Raw slope SE (umol/hr)": "Slope SE (umol/hr)",
                                         "Raw slope %RSE": "Slope %RSE",
                                         "Raw slope 95% CI (umol/hr)": "Slope 95% CI (umol/hr)",
                                         "_Most Common Slope": "Most Common Slope (umol/hr)",
                                        })
        
        # Channels without enough rolling windows cannot have a
        # Most Common Slope calculated.
        df_show["Most Common Slope (umol/hr)"] = (df_show["Most Common Slope (umol/hr)"].apply(
                                                                                               lambda x: "Not enough rolling windows"
                                                                                               if pd.isna(x)
                                                                                               else round(float(x), 3)
                                                                                              )
                                                 )

        if "Most Common Slope (umol/hr)" in df_show.columns:
            # Place the Most Common Slope immediately beside the individual
            # sliding-window slope so the comparison is easy to scan.
            common_slope = df_show.pop("Most Common Slope (umol/hr)")
            slope_position = df_show.columns.get_loc("Slope (umol/hr)") + 1
            df_show.insert(slope_position, "Most Common Slope (umol/hr)", common_slope)

        # If you don't want to show all the columns in the df, you can
        # put some in a side bar.
        
        # Define columns to hide
        hidden_cols = [
                        "intercept",
                        "SSR",
                        "Rate Difference",
                        "Window"
                       ]
                           
        # Build AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_show)
        
        # Apply hide=True to any columns in the hidden list
        # Hide selected columns
        for col in hidden_cols:
            if col in df_show.columns:
                gb.configure_column(col, hide=True)
        
        # Enable column selection sidebar
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_side_bar(columns_panel=True)
       
        # Default settings must be configured before gb.build()
        gb.configure_default_column(
                                    resizable=True,
                                    width=110,
                                    minWidth=80,
                                    maxWidth=150
                                   )
        
        # Specific column widths
        gb.configure_column(
                            "Filename",
                            width=205,
                            minWidth=180,
                            maxWidth=300
                           )
        
        gb.configure_column(
                            "Coral ID",
                            width=110,
                            minWidth=100,
                            maxWidth=150
                           )
        
        gb.configure_column(
                            "Channel",
                            width=90,
                            minWidth=80,
                            maxWidth=110
                           )
        
        gb.configure_column(
                            "Window",
                            width=130,
                            minWidth=110,
                            maxWidth=150
                           )
        
        gb.configure_column(
                            "Start",
                            width=90,
                            minWidth=80,
                            maxWidth=105
                           )
        
        gb.configure_column(
                            "End",
                            width=90,
                            minWidth=80,
                            maxWidth=105
                           )
        
        gb.configure_column(
                            "Slope (umol/hr)",
                            width=150,
                            minWidth=125,
                            maxWidth=220
                           )

        gb.configure_column(
                            "Most Common Slope (umol/hr)",
                            width=235,
                            minWidth=210,
                            maxWidth=320
                           )
        
        gb.configure_column(
                            "R2",
                            width=75,
                            minWidth=65,
                            maxWidth=90
                           )
        
        gb.configure_column(
                            "Slope SE (umol/hr)",
                            width=175,
                            minWidth=135,
                            maxWidth=300
                           )
        
        gb.configure_column(
                            "Slope %RSE",
                            width=160,
                            minWidth=105,
                            maxWidth=200
                           )
        
        gb.configure_column(
                            "Slope 95% CI (umol/hr)",
                            width=200,
                            minWidth=155,
                            maxWidth=300
                           )
        
        # These are hidden, but sizing them is harmless
        gb.configure_column(
                            "intercept",
                            width=110,
                            minWidth=95,
                            maxWidth=130,
                            hide=True
                           )
        
        gb.configure_column(
                            "SSR",
                            width=90,
                            minWidth=80,
                            maxWidth=110,
                            hide=True
                           )

        # Build options after all configuration is complete
        grid_options = gb.build()
        
        # Remove filters panel manually from the built options
        if "sideBar" in grid_options and "toolPanels" in grid_options["sideBar"]:
            grid_options["sideBar"]["toolPanels"] = [
                                                    panel for panel in grid_options["sideBar"]["toolPanels"]
                                                    if panel.get("id") != "filters"
                                                    ]
            
        # Disable Row Groups, Values, and Pivot options
        for panel in grid_options["sideBar"]["toolPanels"]:
            if panel.get("id") == "columns":
                panel["toolPanelParams"] = {
                                            "suppressRowGroups": True,
                                            "suppressValues": True,
                                            "suppressPivots": True,
                                            "suppressPivotMode": True,
                                           }
        
        # Dynamic height calculation with +1 for the header
        num_rows = len(df_show)
        visible_rows = min(num_rows, max_rows)
        height = (visible_rows + 1) * base_row_height
    
        grid_response = AgGrid(
                                df_show,
                                gridOptions=grid_options,
                                update_on=["selectionChanged"],
                                height=height,
                                allow_unsafe_jscode=True,
                                enable_enterprise_modules=True,
                                key=grid_key
                             )
        
        # Remove a phantom column
        df_export = df_show.drop(columns=["::auto_unique_id::"], errors="ignore")
        
        # Download raw values
        rolling_reg_csv = df_export.to_csv(index=False).encode("utf-8-sig")
        
        st.download_button(
                            label="Download Rolling Regression Results",
                            data=rolling_reg_csv,
                            file_name="rolling_regression_results.csv",
                            mime="text/csv",
                          )
        
        # Get the data from a row 
        selected = grid_response["selected_rows"]
        
        if isinstance(selected, pd.DataFrame) and not selected.empty:
            sel_filename = selected["Filename"][0]
            sel_channel = selected["Channel"][0]
            start, end = selected["Start"][0], selected["End"][0]
            slope, intercept = selected["Slope (umol/hr)"][0], selected["intercept"][0]
    
            # Always rebuild the trace from current data
            x_sel, y_sel = final_x_y_valid_dict[sel_filename][sel_channel]
            
            max_points = 100
            if len(x_sel) > max_points:
                idx = np.linspace(0, len(x_sel)-1, max_points).astype(int)
                x_plot, y_plot = x_sel[idx], y_sel[idx]
            else:
                x_plot, y_plot = x_sel, y_sel
            
            channel_trace = go.Scatter(
                                        x=x_plot,
                                        y=y_plot,
                                        mode="lines",
                                        name=sel_channel,
                                        line=dict(color="blue", width=2),
                                        opacity=0.8
                                      )
            
            # Build figure dynamically for the selected window
            fig = go.Figure()
            fig.add_trace(channel_trace)
    
            # Highlight regression line for selected window (dynamic every time)
            x_sel, y_sel = final_x_y_valid_dict[sel_filename][sel_channel] 
            
            mask = (x_sel >= start) & (x_sel <= end)
            # Convert selected x values from seconds to hours for calculating the fitted line
            x_fit_sec = x_sel[mask]
            x_fit_hr = x_fit_sec / 3600
            
            fig.add_trace(go.Scatter(
                                    x=x_fit_sec,
                                    y=float(slope) * x_fit_hr + float(intercept),
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
                                        title="O<sub>2</sub> (μmol)",
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
                            showlegend = False,
                            width=900,
                            height=400 + 100 * ((len(final_x_y_valid_dict[sel_filename]) - 1) // 2),
                            margin=dict(l=80, r=40, t=60, b=60)
                           )
    
            st.plotly_chart(fig, use_container_width=True)
    
    def download_regression(self, all_subset_regs):
        """Create a download button for regressions of the data after the user
        adjusts the x-axis.
        Displays the data in a dynamic data editor table. The table will update
        the blank corrected slope when the user selects a blank file to be subtracted
        from each row of the table.
        """
        
        # Block to update all_subset_regs with blank corrected slope values
        
        # Create a unique ID for the current analysis based on the analyzed filenames
        # Sorting prevents the same files in a different order from producing a different ID
        current_analysis_id = "|".join(sorted(all_subset_regs.keys()))
        
        # If the analyzed files changed, clear old editor/session state
        # Otherwise results table won't change when new physio channels are selected for analysis
        if st.session_state.get("regression_analysis_id") != current_analysis_id:
            
            # Create and add an id to session state
            st.session_state.regression_analysis_id = current_analysis_id
            
            # Clear all stale dictionaries and blank file selections
            # so old results aren't shown 
            st.session_state.pop("all_subset_regs", None)
            st.session_state.pop("blank_subtracted_selections", None)
            # Clear the data editor table
            st.session_state.pop("regression_results_editor", None)
    
        # Use updated regression dictionary if blank corrections were applied
        if "all_subset_regs" in st.session_state:
            all_subset_regs = st.session_state.all_subset_regs
    
        rows = []
    
        # Create list of available blank files
        blank_files = []
    
        # Loop through files and channels to identify files labeled as blanks
        for filename, channels in all_subset_regs.items():
            for ch, results in channels.items():
    
                # Find the coral ID for this file/channel
                coral_id = str(results.get("Coral ID", "")).strip().lower()
                
                # If any channel in this file is labeled as blank, add the filename to the blank-file list
                # Then break and go to next file
                if coral_id == "blank":
                    blank_files.append(filename)
                    break
    
        # Remove duplicate entries
        blank_files = sorted(set(blank_files))
        
        # Keep placeholder first so it appears as the selectbox default
        blank_file_options = ["Select Blank File"] + blank_files
    
        # Initialize blank selections dict in session_state if it doesn't exist
        if "blank_subtracted_selections" not in st.session_state:
            st.session_state.blank_subtracted_selections = {}
    
        # Get current regression/statistical values for this file/channel
        for filename, channels in all_subset_regs.items():
            for ch, results in channels.items():
                
                slope = results.get("slope (umol/hr)", np.nan)
                slope_se = results.get("slope stderr (umol/hr)", np.nan)
                corrected_slope_se = results.get("Corrected slope stderr (umol/hr)", np.nan)
                slope_CI = results.get("slope 95% CI (umol/hr)", np.nan)
                corrected_slope_CI = results.get("Corrected slope 95% CI (umol/hr)", np.nan)
                n_points = results.get("Total Pts", np.nan)
                unique_pts = results.get("Unique Pts", np.nan)
                r2 = results.get("R2", np.nan)
                slope_rse = results.get("slope % RSE", np.nan)
                blank_corrected_slope = results.get("blank-corrected slope (umol/hr)", np.nan)
                blank_corrected_slope_se = results.get("blank-corrected slope SE (umol/hr)", np.nan)
                blank_corrected_slope_CI = results.get("blank-corrected 95% CI (umol/hr)", np.nan)
                blank_corrected_lag_se = results.get("blank-corrected Lag-adjusted SE (umol/hr)", np.nan)
                blank_corrected_lag_CI = results.get("blank-corrected Lag-adjusted 95% CI (umol/hr)", np.nan)
    
                # Create a unique key for each row of the table so each one can have
                # its own blank file subtracted
                row_key = f"{filename}__{ch}"
    
                # Get the previously selected blank filename, if one exists in the results dict
                # Returns None if "Blank Subtracted" is not present
                default_blank = results.get("Blank Subtracted")
                
                # If no valid blank filename exists, use the selectbox placeholder
                if default_blank in [None, "", "Blank File"] or pd.isna(default_blank):
                    default_blank = "Select Blank File"
                
                # Initialize this row's blank-file selection in session_state if it has not been stored yet
                if row_key not in st.session_state.blank_subtracted_selections:
                    st.session_state.blank_subtracted_selections[row_key] = default_blank
    
                # Retrieve the currently selected blank file for this row
                blank_subtracted = st.session_state.blank_subtracted_selections[row_key]
    
                # Calculate lag-adjusted slope %RSE using the lag-adjusted SE and the original slope 
                if pd.notna(corrected_slope_se) and pd.notna(slope) and float(slope) != 0:
                    corrected_slope_rse = 100 * float(corrected_slope_se) / abs(float(slope))
                    
                else:
                    corrected_slope_rse = np.nan
    
                rows.append({
                            # Metadata
                            "Filename": filename,
                            "Blank Subtracted": blank_subtracted,
                            "Channel": ch,
                            "Coral ID": results.get("Coral ID", np.nan),
                            "Volume (mL)": results.get("Volume (mL)", np.nan),
                            "Start Time": results.get("Start Time", np.nan),
                            "Stop Time": results.get("Stop Time", np.nan),
                            "Total Pts": n_points,
                            "Unique Pts": unique_pts,
                            "Noise %": results.get("Noise %", np.nan),
                            "Rate Shift": results.get("Rate Shift", ""),
    
                            # Hidden row key for updating session_state
                            "row_key": row_key,
    
                            # Uncorrected slope info
                            "Raw slope (umol/hr)": slope,
                            "R2": r2,
                            "Raw slope SE (umol/hr)": slope_se,
                            "Raw slope % RSE": slope_rse,
                            "Raw slope 95% CI (umol/hr)": slope_CI,
    
                            # Blank corrected slope info
                            "Blank-corrected slope (umol/hr)": blank_corrected_slope,
                            "Blank-corrected SE (umol/hr)": blank_corrected_slope_se,
                            "Blank-corrected 95% CI (umol/hr)": blank_corrected_slope_CI,
    
                            # Lag-adjusted slope
                            "Lag-adjusted SE (umol/hr)": corrected_slope_se,
                            "Lag-adjusted % RSE": corrected_slope_rse,
                            "Lag-adjusted 95% CI (umol/hr)": corrected_slope_CI,
    
                            # Blank and lag adjusted slope stats
                            "Blank-corrected, Lag-adjusted SE (umol/hr)": blank_corrected_lag_se,
                            "Blank-corrected, Lag-adjusted 95% CI (umol/hr)": blank_corrected_lag_CI,
                            
                            # Add treatment
                            "Treatment":results.get("Treatment", np.nan),
    
                            # "Spike Score": results.get("Spike Score", np.nan),
                            # "slope pval": results.get("slope pval", np.nan),
                            # "SSR": results.get("squared residuals", np.nan),
                            # "System Lag (Pts)": results.get("System Lag (Pts)", np.nan),
                          })
    
        # Raw dataframe for CSV/download
        df_raw = pd.DataFrame(rows)
        
        # Create warnings column for values outside of thresholds
        # Initialize empty warnings column
        df_raw["Warnings"] = ""
        
        # Low R2
        df_raw.loc[
                  (df_raw["R2"].notna()) & (df_raw["R2"] < 0.95),
                  "Warnings"
                  ] += "Low R2; "
        
        # High % RSE
        df_raw.loc[
                  (df_raw["Lag-adjusted % RSE"].notna()) & (df_raw["Lag-adjusted % RSE"] > 50),
                  "Warnings"
                  ] += "High % RSE; "
        
        # High Noise
        df_raw.loc[
                  (df_raw["Noise %"].notna()) & (df_raw["Noise %"] > 20),
                  "Warnings"
                  ] += "High Noise; "
        
        # Two sustained 5-minute windows support substantially different rates.
        df_raw.loc[
                  df_raw["Rate Shift"].eq("Yes"),
                  "Warnings"
                  ] += "Possible Rate Shift; "
        
        # Clean up trailing separators and replace empty with NA
        df_raw["Warnings"] = (
                              df_raw["Warnings"]
                              .str.rstrip("; ")
                              .replace("", "")
                             )
        
        # Formatted copy rounded for display only
        df_display = df_raw.copy()
    
        # Round columns for display only
        round_5_cols = [
                        "Raw slope (umol/hr)",
                        "Raw slope SE (umol/hr)",
                        "Lag-adjusted SE (umol/hr)",
                        "Blank-corrected slope (umol/hr)",
                        "Blank-corrected SE (umol/hr)",
                        "Blank-corrected, Lag-adjusted SE (umol/hr)"
                       ]
    
        # Rounding
        for col in round_5_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                                                        lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
                                                       )
    
        if "R2" in df_display.columns:
            df_display["R2"] = df_display["R2"].apply(
                                                      lambda x: round(float(x), 3) if pd.notna(x) else np.nan
                                                     )
    
        if "Raw slope % RSE" in df_display.columns:
            df_display["Raw slope % RSE"] = df_display["Raw slope % RSE"].apply(
                                                                                lambda x: round(float(x), 2) if pd.notna(x) else np.nan
                                                                               )
    
        if "Lag-adjusted % RSE" in df_display.columns:
            df_display["Lag-adjusted % RSE"] = df_display["Lag-adjusted % RSE"].apply(
                                                                                      lambda x: round(float(x), 2) if pd.notna(x) else np.nan
                                                                                     )
    
        if "Unique Pts" in df_display.columns:
            df_display["Unique Pts"] = df_display["Unique Pts"].apply(
                                                                      lambda x: round(float(x), 1) if pd.notna(x) else np.nan
                                                                     )
    
        if "Start Time" in df_display.columns:
            df_display["Start Time"] = df_display["Start Time"].apply(
                                                                      lambda x: round(float(x), 3) if pd.notna(x) else np.nan
                                                                     )
    
        if "Stop Time" in df_display.columns:
            df_display["Stop Time"] = df_display["Stop Time"].apply(
                                                                    lambda x: round(float(x), 3) if pd.notna(x) else np.nan
                                                                   )
    
        if "Noise %" in df_display.columns:
            df_display["Noise %"] = df_display["Noise %"].apply(
                                                                lambda x: round(float(x), 1) if pd.notna(x) else np.nan
                                                               )
    
        # Show formatted table
        st.markdown(
                    "<p style='color: Blue; font-size: 24px; margin: 0;'>Final Regression Results</p>",
                    unsafe_allow_html=True
                   )
    
        edited_df = st.data_editor(
                                    df_display,
                                    use_container_width=True,
                                    num_rows="static",
                                    hide_index=True,
                                    key="regression_results_editor",
                                    disabled=[
                                                col for col in df_display.columns
                                                if col != "Blank Subtracted"
                                             ],
                                    column_config={
                                                    "Blank Subtracted": st.column_config.SelectboxColumn(
                                                                                                         "Blank Subtracted",
                                                                                                         options=blank_file_options,
                                                                                                         required=True
                                                                                                         ),
                                                                                                         "row_key": None,
                                                  }
                                  )
    
        # Use a button to trigger calculations and update the data editor table
        if st.button("Apply Blank Corrections"):
    
            # Store edited blank selections
            for index, row in edited_df.iterrows():
                # Retrieve row_key and selected blank for each row
                row_key = row["row_key"]
                selected_blank = row["Blank Subtracted"]
                # Store row_key and selected blank in session state
                st.session_state.blank_subtracted_selections[row_key] = selected_blank
    
            # Apply edited blank selections and recalculate corrections
            # Pass all_subset_regs and the data editor table
            all_subset_regs = self.apply_edited_blank_selections(
                                                                 all_subset_regs=all_subset_regs,
                                                                 edited_df=edited_df,
                                                                )
    
            # Store updated dictionary so table rebuilds with corrected values
            st.session_state.all_subset_regs = all_subset_regs
    
            st.rerun()
    
        # Download results
        df_download = edited_df.drop(columns=["row_key"], errors="ignore")
        csv = df_download.fillna("NA").to_csv(index=False).encode("utf-8-sig")
    
        st.download_button(
                           label="Download Regression Results",
                           data=csv,
                           file_name="regression_results.csv",
                           mime="text/csv",
                          )
    
    
    def apply_edited_blank_selections(self,
                                      all_subset_regs,
                                      edited_df,
                                      slope_key="slope (umol/hr)",
                                      se_key="slope stderr (umol/hr)",
                                      corrected_se_key="Corrected slope stderr (umol/hr)",
                                      coral_id_key="Coral ID",
                                      blank_id="Blank",
                                      ci_multiplier=1.96,
                                     ):
        """Apply user-edited Blank Subtracted choices and recalculate blank corrections."""
    
        # Build lookup of blank filename -> blank regression data
        blank_lookup = {}
    
        # Loop through files and channels
        for filename, channels in all_subset_regs.items():
            # Loop through regression results of each channel
            for ch, results in channels.items():
                # Retrieve coral_id
                coral_id = str(results.get(coral_id_key, "")).strip().lower()
                # If coral id is a blank, return results
                if coral_id == blank_id.lower():
                    blank_lookup[filename] = results
                    break
    
        # Apply each edited row of the data editor table
        for _, row in edited_df.iterrows():
    
            filename = row["Filename"]
            ch = row["Channel"]
            blank_filename = row["Blank Subtracted"]
            
            # Safety checks-skip to next row if these don't exist
            if filename not in all_subset_regs:
                continue
    
            if ch not in all_subset_regs[filename]:
                continue
            
            # Retreive the regression data for the current filename from all_subset_regs
            results = all_subset_regs[filename][ch]
            results["Blank Subtracted"] = blank_filename
    
            # Set all values to None if no valid blank selected
            if blank_filename == "Select Blank File" or blank_filename not in blank_lookup:
                self.clear_blank_results(results)
                continue
            
            # Retrieve the regression results for the blank selected
            blank_data = blank_lookup[blank_filename]
            # Get the raw slopes from data editor table
            sample_slope = results.get(slope_key)
            blank_slope = blank_data.get(slope_key)
            
            # If this row is itself a blank being corrected by its own file,
            # report zero slope and no propagated uncertainty.
            coral_id = str(results.get(coral_id_key, "")).strip().lower()
            
            if coral_id == blank_id.lower() and filename == blank_filename:
                self.clear_blank_results(results)
                results["blank-corrected slope (umol/hr)"] = 0.0
                continue
    
            if sample_slope is None or blank_slope is None:
                self.clear_blank_results(results)
                continue
    
            # Blank-corrected slope
            bc_slope = sample_slope - blank_slope
            results["blank-corrected slope (umol/hr)"] = bc_slope
    
            # Standard SE error propagation
            sample_se = results.get(se_key)
            blank_se = blank_data.get(se_key)
    
            if sample_se is not None and blank_se is not None:
                bc_se = math.sqrt(sample_se**2 + blank_se**2)
    
                results["blank-corrected slope SE (umol/hr)"] = bc_se
                results["blank-corrected CI Low (umol/hr)"] = bc_slope - ci_multiplier * bc_se
                results["blank-corrected CI High (umol/hr)"] = bc_slope + ci_multiplier * bc_se
                results["blank-corrected 95% CI (umol/hr)"] = (
                                                                f"{results['blank-corrected CI Low (umol/hr)']:.3f} – "
                                                                f"{results['blank-corrected CI High (umol/hr)']:.3f}"
                                                                )
                
            else:
                results["blank-corrected slope SE (umol/hr)"] = None
                results["blank-corrected CI Low (umol/hr)"] = None
                results["blank-corrected CI High (umol/hr)"] = None
                results["blank-corrected 95% CI (umol/hr)"] = None
    
            # Lag-adjusted SE propagation
            sample_corr_se = results.get(corrected_se_key)
            blank_corr_se = blank_data.get(corrected_se_key)
    
            if sample_corr_se is not None and blank_corr_se is not None:
                bc_corr_se = math.sqrt(sample_corr_se**2 + blank_corr_se**2)
    
                results["blank-corrected Lag-adjusted SE (umol/hr)"] = bc_corr_se
                results["blank-corrected Lag-adjusted CI Low (umol/hr)"] = (
                                                                             bc_slope - ci_multiplier * bc_corr_se
                                                                            )
                results["blank-corrected Lag-adjusted CI High (umol/hr)"] = (
                                                                              bc_slope + ci_multiplier * bc_corr_se
                                                                             )
                results["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = (
                                                                             f"{results['blank-corrected Lag-adjusted CI Low (umol/hr)']:.3f} – "
                                                                             f"{results['blank-corrected Lag-adjusted CI High (umol/hr)']:.3f}"
                                                                            )
            
            else:
                results["blank-corrected Lag-adjusted SE (umol/hr)"] = None
                results["blank-corrected Lag-adjusted CI Low (umol/hr)"] = None
                results["blank-corrected Lag-adjusted CI High (umol/hr)"] = None
                results["blank-corrected Lag-adjusted 95% CI (umol/hr)"] = None
    
        return all_subset_regs
        
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
 
    def kde(self, 
            rolling_df: pd.DataFrame,
            slope_col: str = "Raw slope (umol/hr)",
            grid_n: int = 400,
            min_r2: float | None = None,   
           ) -> pd.DataFrame:
        
        """
        Summarizes the rolling-slope distribution for each Filename x Channel.
        
        Most Common Slope is the dominant KDE mode. Rate Variability is the
        standard deviation of all rolling slopes relative to the modal slope.
        Slope Assessment describes the degree of variability among the rolling
        slopes without assigning a biological cause to that variability.
        """
      
        if rolling_df is None or getattr(rolling_df, "empty", True):
            st.warning("Not enough data exists to calculate the most common slope.")
            return pd.DataFrame()
    
        df = rolling_df.copy()
       
        # Numeric coercion
        df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    
        # Optional R2 filtering is retained for backwards compatibility but is
        # not used by default.
        if min_r2 is not None:
            df = df[df["R2"].notna() & (df["R2"] >= float(min_r2))]
    
        rows = []
    
        for (fname, ch), g in df.groupby(["Filename", "Channel"], sort=False):
            slopes = g[slope_col].dropna().to_numpy(dtype=float)
           
            metrics = self.rolling_rate_metrics(slopes, grid_n=grid_n)

            # rolling_rate_metrics() determines whether enough windows
            # exist to calculate a most common slope.
            if metrics["assessment"] == "Not enough rolling windows":
                rows.append({
                            "Filename": fname,
                            "Channel": ch,
                            "Most Common Slope (umol/hr)": "Not enough rolling windows",
                            "Rate Variability (%)": "Not enough rolling windows",
                            "Slope Assessment": "Not enough rolling windows",
                           })
                continue

            variability = metrics["variability_percent"]

            if np.isfinite(variability):
                variability_display = variability
                
            else:
                variability_display = "Could not calculate — slope near 0"
            
            rows.append({
                        "Filename": fname,
                        "Channel": ch,
                        "Most Common Slope (umol/hr)": metrics["mode"],
                        "Rate Variability (%)": variability_display,
                        "Slope Assessment": metrics["assessment"],
                       })
      
        summary = pd.DataFrame(rows)
    
        if summary.empty:
            st.warning("No KDE summary available (not enough rolling windows per channel after filtering).")
            return summary
    
        # Rounding for display
        summary = summary.copy()
        
        for col, decimals in [("Most Common Slope (umol/hr)", 3), ("Rate Variability (%)", 1),]:
            summary[col] = summary[col].apply(
                                              lambda x, d=decimals: round(float(x), d)
                                              if isinstance(x, (int, float, np.number)) and np.isfinite(x)
                                              else x
                                             )
    
        return summary
    
    def kde_render(self,
                   summary,
                   title: str = "Rolling Slope Summary",
                   button_text: str = "Download Rolling Slope Summary",
                   file_name: str = "rolling_slope_summary.csv",
                   
                   # Display controls
                   sort_by = None
                  ):
        
        # st.write('')
        st.divider()
        
        # Display 
        st.markdown(
                    f"<p style='color: dodgerblue; font-size: 20px; margin: 0;'><b>{title}</b></p>",
                    unsafe_allow_html=True
                   )
        
        st.write("")
        
        # Sort
        if sort_by in summary.columns:
            summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True) 
            
        summary_str = summary.astype(str)
    
        st.dataframe(
                     summary_str,
                     use_container_width=False,
                     hide_index=True,
                     column_config={
                                    "Slope Assessment": st.column_config.TextColumn("Slope Assessment", 
                                                                                    width="large"
                                                                                    )
                                   }
                    )
        
        # Download
        csv = summary.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
                            label=button_text,
                            data=csv,
                            file_name=file_name,
                            mime="text/csv",
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
                
class About():
    
    def __init__(self):
        
        pass
    
    def about_tab(self):
        
        st.write('')
     
        st.markdown(f"<p style='color: blue; \
                      font-size: 24px; \
                      margin: 0;'>ResPy Users Guide</p>",
                      unsafe_allow_html=True)
        
        st.write('')
        
        col1, col2 = st.columns([2,1])
        
        with col1:
        
            with st.expander('🪸 File Formatting'):
                
                st.markdown("""
                ### File formatting
                - **Raw Data Files** → Raw data files may be `.csv` or `.txt` format. 
                ResPy needs to parse the filename in order to retrieve information from the master physio metadata file.  
                  The filename should be structured as:
                
                  `date_Run#_Light/Dark.txt`
                
                  For example: `08_02_25_Run1A_Light.txt` or `08_02_25_Run12B_Dark.txt`.
                
                - **Metadata Files** → Metadata files must be `.csv` files. These must have the following columns written exactly as shown (`start_time` and `treatment` are not necessary, but can aid in troubleshooting):
                
                  - `start_time`
                  - `date`
                  - `filename`
                  - `run`
                  - `group`
                  - `light_dark`
                  - `channel`
                  - `replicate`
                  - `volume_mL`
                  - `coral_ID`
                  - `treatment`
                  
                  The replicate number should only change if a run on the same fragments is restarted under identical conditions.
                """
                ,
                unsafe_allow_html=True
                )
                  
                st.write('')
                
                # Create a downloadable template
                template_df = pd.DataFrame([{
                                            "start_time": "08:00",
                                            "date": "08_02_25",
                                            "filename": "08_02_25_Run1A_Light",
                                            "run": 1,
                                            "group": "A",
                                            "light_dark": "LIGHT",
                                            "channel": "ch1",
                                            "replicate": 1,
                                            "volume_mL": 100,
                                            "coral_ID": "C1",
                                            "treatment": "control"
                                           }])
                
                st.download_button(
                                    label="📥 Download Physio Master Metadata Template",
                                    data=template_df.to_csv(index=False),
                                    file_name="metadata_template.csv",
                                    mime="text/csv"
                                  )
                
        
            with st.expander('🐟 ResPy Statistics Tutorial'):
            
                # Use this for cloud
                stats_files = [
                                BASE_DIR + "Slide1.png",
                                BASE_DIR + "Slide2.png",
                                BASE_DIR + "Slide3.png",
                                BASE_DIR + "Slide4.png",
                              ]
                
                # Use this for local host
                # stats_files = [
                #                 BASE_DIR + "/Slide1.png",
                #                 BASE_DIR + "/Slide2.png",
                #                 BASE_DIR + "/Slide3.png",
                #                 BASE_DIR + "/Slide4.png",  
                #               ]
        
                for slide in stats_files:
                    st.image(slide, use_container_width=True)
                    # st.write('')
                    st.divider()
                    
            with st.expander('🦈 Suggested Workflow'):
                
                # Use this for cloud
                workflow_files = [
                                  BASE_DIR + "Slide5.png",
                                  BASE_DIR + "Slide6.png",
                                  BASE_DIR + "Slide7.png",                  
                                 ]
                
                # Use this for local host
                # workflow_files = [BASE_DIR + "/Slide5.png",
                #                   BASE_DIR + "/Slide6.png",
                #                   BASE_DIR + "/Slide7.png",
                #                   ]
                

                for slide in workflow_files:
                    st.image(slide, use_container_width=True)
                    # st.write('')
                    st.divider()
                    
    def results_interpretation(self):
        
        with st.expander('🪸 Results Definitions'):
            
            st.markdown(
                        """
                        ### 📊 What does it all mean? 
                        
                        **Metadata**
                        - **Filename** → Name of the uploaded data file.
                        - **Channel** → Sensor channel used for the measurement.
                        - **Coral ID** → Identifier for the coral fragment.
                        - **Volume (mL)** → Volume of the respiration chamber.
                        - **Start Time / Stop Time** → Time window used for slope calculation.
                        - **Total Pts** → Total number of data points in the selected window.
                        - **Unique Pts** → Estimated number of *independent* points (accounts for autocorrelation).
                        - **Noise %** → Estimated noise level in the signal relative to the slope.
                        
                        ---
                        
                        **Raw (Uncorrected) Slope**
                        - **Raw slope (µmol/hr)** → Rate of oxygen change over time (before any corrections).
                        - **R²** → Goodness of fit of the linear regression (closer to 1 = more linear).
                        - **Raw slope SE (µmol/hr)** → Standard error of the slope (uncertainty of the estimate).
                        - **Raw slope % RSE** → Relative standard error (SE expressed as % of slope magnitude).
                        - **Raw slope 95% CI (µmol/hr)** → Confidence interval for the slope (uncertainty range).
                        
                        ---
                        
                        **Blank-Corrected Slope**
                        - **Blank-corrected slope (µmol/hr)** → Slope after subtracting background (blank) signal.
                        - **Blank-corrected SE (µmol/hr)** → Standard error after blank correction.
                        - **Blank-corrected 95% CI (µmol/hr)** → Confidence interval after blank correction.
                        
                        ---
                        
                        **Lag-Adjusted (Autocorrelation-Corrected)**
                        - **Lag-adjusted SE (µmol/hr)** → Standard error corrected for autocorrelation (e.g., Newey–West).
                        - **Lag-adjusted % RSE** → Relative standard error after lag correction.
                        - **Lag-adjusted 95% CI (µmol/hr)** → Confidence interval accounting for time-series dependence.
                        
                        ---
                        
                        **Blank + Lag Adjusted**
                        - **Blank-corrected, Lag-adjusted SE (µmol/hr)** → Final uncertainty accounting for both blank correction and autocorrelation.
                        - **Blank-corrected, Lag-adjusted 95% CI (µmol/hr)** → Most reliable confidence interval for the corrected slope.
                        
                        ---
                        
                        💡 *Tip:*  
                        - Use **lag-adjusted values** when autocorrelation is present.  
                        - Use **blank-corrected values** when background drift or system respiration is significant.  
                        - The **final (blank + lag adjusted)** values are typically the most reliable for comparisons.
                        
                        ### Please see the About tab for more information.
                        """
                        )
        
                 
# Run 
if __name__ == '__main__':
   
    # Get the path relative to the current file (inside Docker container)
    BASE_DIR = os.path.dirname(__file__)
        
    st.session_state['logo'] = os.path.join(BASE_DIR, 'mote_logo.png')
    
    #----------------------------------------------------------------------
    # Use this for cloud-Renders image in header
    # Load image for header
    logo_img = Image.open(st.session_state['logo'])

    # Load smaller version of image for favicon
    favicon_img = Image.open(os.path.join(BASE_DIR, 'mote_logo.png')).resize((32, 32))
    
    #----------------------------------------------------------------------
    # Use this for local machine
    # logo_img = Image.open(st.session_state['logo'])
    # favicon_img = logo_img.resize((32, 32))
    #----------------------------------------------------------------------
    
    # Page config
    st.set_page_config(layout = "wide", 
                       page_title = 'Mote', 
                       page_icon = favicon_img,
                       initial_sidebar_state="auto", 
                       menu_items = None)
    
    
    # Call Flow_Control class that makes all calls to other classes and methods
    obj1 = Flow_Control()
    all_calls = obj1.all_calls()
