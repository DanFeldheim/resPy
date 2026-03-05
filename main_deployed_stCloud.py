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
from statsmodels.graphics.tsaplots import plot_acf
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
                                "snow":False,
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
    
            # Create an upload button to load file names to be uploaded for the 
            # raw data and master physio metadata spreadsheet
            photo_files, master_file = load_files.upload()
            
            # st.write('')
            st.write('')
            st.divider()
            
            if master_file and photo_files:
              
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>View Your Dataset</p>",
                      unsafe_allow_html=True)
                    
                st.write('')
                
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
                
                # Matches run, run letter, and light_dark to the physiomaster spreadsheet.
                # to extract volume and coral ID from the physiomaster file.
                # Debug will activate a series of error messages for missing data.
                edited_raw_dfs_dict = load_files.find_volume(
                                                             edited_raw_dfs_dict, 
                                                             master_file_df,
                                                             debug = False
                                                             ) 
            
                #--------------------------------------------------------------------------------------
                # Congratulations! You now have a dict with channel data, volume, and coral ID
                    
                # Loop through  edited_raw_dfs_dict and do
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
                                # st.write(volume, coral_id)
                            
                                # Apply corrections directly to the DataFrame column so that
                                # edited_raw_dfs_dict is changed at all points downstream
                                df[ch] = df[ch] * 0.99975 * (volume / 1000)
                                
                                # Extract corrected y values for regression
                                y = df[ch].values
                    
                                # Fit regression
                                regress, residuals = plot_data.fit_regression(x, y)
                                
                                rmse, noise_percent, rse_percent, zmax = self.noise_calc(residuals, 
                                                                                         regress, 
                                                                                         duration)
                    
                                # Estimate maxlags as an indication of residual autocorrelation
                                maxlag = plot_data.estimate_maxlags(residuals)
                    
                                # Store results for all channels and filenames in one dict
                                all_regression_results.setdefault(filename, {})[ch] = {
                                                                                # Unpack regression stats
                                                                                **regress,                 
                                                                                "Start Time": start_time,
                                                                                "Stop Time": stop_time,
                                                                                "MaxLag": maxlag, 
                                                                                "Volume (mL)":volume,
                                                                                "Coral ID":coral_id,
                                                                                "RMSE":rmse,
                                                                                "Noise %":noise_percent,
                                                                                "slope % RSE":rse_percent,
                                                                                "Zmax":zmax
                                                                                }
                                
                            except Exception as e:
                                
                                # If either "Volume (mL)" or "Coral ID" is missing for this channel,
                                # record a placeholder and continue
                                volume = "Not found"
                                coral_id = "Not found"
                                all_regression_results.setdefault(filename, {})[ch] = {
                                                                                        "Error":e,
                                                                                        "Start Time": start_time,
                                                                                        "Stop Time": stop_time,
                                                                                        "MaxLag": None,
                                                                                        "Volume (mL)": volume,
                                                                                        "Coral ID": coral_id,
                                                                                        "RMSE":None,
                                                                                        "Noise %":None,
                                                                                        "slope % RSE":None,
                                                                                        "Zmax":None
                                                                                       }
                                continue
                
                st.write("")
                st.write("")
                st.markdown(f"<p style='color: Blue; \
                      font-size: 24px; \
                      margin: 0;'>Preview Linear Regression Results</p>",
                      unsafe_allow_html=True)
                
                # Show results in aggrid table and return selected rows
                selected_rows = plot_data.agtable(all_regression_results)
              
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
                    for filename, vals in subset_data.items():
                       
                        df = vals['data']

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
                            
                            # Get the start/stop times for the x-axis
                            start_time = float(min(x_valid))
                            stop_time = float(max(x_valid))
                            duration = stop_time - start_time
                           
                            # Do the regression and get maxlags
                            regress, residuals = plot_data.fit_regression(x_valid, y_valid)
                            maxlag = plot_data.estimate_maxlags(residuals)
                            
                            # Calculate noise % and other noise parameters
                            rmse, noise_percent, rse_percent, zmax = self.noise_calc(residuals, 
                                                                                     regress, 
                                                                                     duration)
                            
                            # Get per-channel metadata 
                            metadata = vals['metadata'].get(channel, {})
                            coral_id = metadata.get('coral_id', 'Unknown')
                            volume = metadata.get('volume', 'Unknown')
                            
                            # Combine results for the current channel in a dict
                            channel_results[channel] = {
                                                        **regress,
                                                        "MaxLag": maxlag,
                                                        "Start Time":start_time,   
                                                        "Stop Time":stop_time,
                                                        "Volume (mL)":volume,
                                                        "Coral ID":coral_id,
                                                        "Noise %": noise_percent,
                                                        "slope % RSE":rse_percent,
                                                        "RMSE":rmse,
                                                        "Zmax":zmax
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
                                                               show_warning = False
                                                              )
                        
                        st.divider()
                    
                    try:
                        # Displays a table of all regression results for selected channels
                        # and a button to download regression results.
                        download_reg = plot_data.download_regression(all_subset_regs)   
                    
                        # Download plots
                        download_plots = plot_data.download_respiration_plots(all_plots)
                        
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
                
                # Subset all_subset_regs
                # Keep these
                keys_to_keep = [
                                "slope (umol/L/s)", 
                                "R2", "slope stderr (umol/L/s)", 
                                "slope 95% CI (umol/L/s)", 
                                "slope pval"
                               ]
                
                # Make a new dictionary for storing corrected regressions
                # Filters all_subset_regs to store just the keys_to_keep
                corrected_regression_dict = {}
                
                # Loop through regression data
                for filename, channels in all_subset_regs.items():
                    
                    # Create an entry for this filename in the dict instantiated above
                    corrected_regression_dict[filename] = {}
                
                    for ch, vals in channels.items():
                        # Create an entry for this channel
                        corrected_regression_dict[filename][ch] = {}
                       
                        # Retrieve a value if it was in the list of keys to keep above
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
                csv = final_corrected_df.to_csv(index=False).encode("utf-8-sig")
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
                                                key=f"win_size_{filename}"  
                                                )
                    st.write('')
                    
                    st.markdown(f"<p style='color: dodgerblue; \
                                  font-size: 20px; \
                                  margin: 0;'>Best Windows Ranked by R<sup>2</sup>.</p>",
                                  unsafe_allow_html=True)
                        
                    st.write('')

                    st.markdown(f"<p style='color: black; \
                                  font-size: 14px; \
                                  margin: 0;'>Select a channel to view plot and fit line.</p>",
                                  unsafe_allow_html=True)
                    
                    st.write('')
                       
                    # Computes regression fits for small windows in the data
                    rolling_reg = plot_data.compute_rolling_regression(final_x_y_valid_dict, window_size)
                    
                    # Displays rolling regression table and plot
                    rolling_reg_plots = plot_data.rolling_reg_ui(final_x_y_valid_dict, rolling_reg)
                    
                    kde_analysis_df = plot_data.kde(rolling_reg, slope_col="Slope (umol/L/s)", tol_sd_mult=1.0)

                else:
                    
                    col1, col2, col3 = st.columns([0.5,6,1])
                    
                    with col2:
                    
                        st.markdown(f"<p style='color: Red; \
                              font-size: 24px; \
                              margin: 0;'>Please go to the Linear Regression tab, upload data, and select channels to analyze.</p>",
                              unsafe_allow_html=True)
                            
                            
                
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
                        ### 🎛️ Noise Analysis
                        - **Residuals Plots** → Residuals should be scattered randomly around the 0 line.
                        A non-random pattern in the residuals indicates autocorrelation or heteroskedasticity.
                        This violates one of the assumptions of linear regression and indicates sluggishness
                        in the experiment-slow mixing or sensor response relative to the sampling rate. 
                        A consequence of residual correlation is that standard errors, confidence intervals, 
                        and p-values are underestimated. 
                        These values can be corrected with a Newey-West correction (see Signal Processing).
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
        slope = regress.get("slope (umol/L/s)", np.nan)
        stderr = regress.get("slope stderr (umol/L/s)", np.nan)
    
        # Zmax: spike/outlier score (largest residual relative to typical residual scale)
        if res.size and np.isfinite(rmse) and rmse > 0:
            zmax = float(np.max(np.abs(res)) / rmse)
    
        # If slope isn't valid, we can't compute Noise% or RSE%
        if not isinstance(slope, (float, int)) or not np.isfinite(slope):
            return rmse, noise_percent, rse_percent, zmax
    
        duration_s = float(duration_s)
    
        # Noise % = RMSE / total signal change across the window
        delta_y = abs(slope) * duration_s
        
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
                                              )
            
        with col2: 
            
            # File uploader that allows a single file
            master_file = st.file_uploader(
                                           "Choose Physio Master File", 
                                           type=["csv", "txt"],  
                                           accept_multiple_files=False, 
                                           )
            
            
        
        
        return uploaded_files, master_file
        
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
    
    def upload_master(self, file):
        
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file, encoding=enc)
                
                return df
            
            except UnicodeDecodeError:
                continue
            
            except Exception as e:
                st.error(f"❌ Error reading CSV file {file.name}: {e}")
                return None
        
    @staticmethod
    @st.cache_data
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
    
    def find_volume(self, edited_raw_dfs_dict, master_file_df, debug=False):
        """
        Takes in data from edited_raw_dfs_dict and retrieves volume and coral ID 
        from master_file_df (the metadata file). Adds these to the edited_raw_dfs_dict
        and returns it. 

        Parameters
        ----------
        edited_raw_dfs_dict : TYPE
            DESCRIPTION.
        master_file_df : TYPE
            DESCRIPTION.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        edited_raw_dfs_dict : TYPE
            DESCRIPTION.

        """

        # Copy for safety
        mdf = master_file_df.copy()
    
        # Normalize master sheet columns by stripping, stringing, datetiming, and inting
        mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce").dt.date
        mdf["filename"] = (
                           mdf["filename"].astype(str).str.strip()
                           .apply(lambda x: os.path.splitext(x)[0])
                           .str.lower()
                          )
        mdf["run"] = pd.to_numeric(mdf["run"], errors="coerce").astype("Int64")
        mdf["channel"] = mdf["channel"].astype(str).str.strip().str.lower()
        mdf["group"] = mdf["group"].astype(str).str.strip().str.upper()
        mdf["light_dark"] = mdf["light_dark"].astype(str).str.strip().str.upper()
    
        # Loop through data files to match data with volumes and coral_IDs from metadata file
        for clean_filename, file_info in edited_raw_dfs_dict.items():
    
            filename = os.path.splitext(str(clean_filename).strip())[0].lower()
            file_date = pd.to_datetime(str(file_info["date"]).strip(), errors="coerce").date()
            file_run = pd.to_numeric(str(file_info["run"]).strip(), errors="coerce")
            file_run = int(file_run) if pd.notna(file_run) else None
            file_group = str(file_info["group"]).strip().upper()
            file_light_dark = str(file_info["light_dark"]).strip().upper()
    
            df = file_info["data"]
            channel_metadata = {}
    
            # Pre-filter once per file (big win)
            file_subset = mdf[
                              (mdf["filename"] == filename) &
                              (mdf["date"] == file_date) &
                              (mdf["run"] == file_run) &
                              (mdf["group"] == file_group) &
                              (mdf["light_dark"] == file_light_dark)
                             ]
    
            for ch in (c for c in df.columns if str(c).lower().startswith("ch")):
                channel = str(ch).strip().lower()
                match = file_subset[file_subset["channel"] == channel]
    
                if match.empty:
                    channel_metadata[ch] = {"volume_mL": None, "coral_ID": None}
                    if debug:
                        st.write(f"❌ No match for {filename} | {ch}")
    
                elif len(match) == 1:
                    channel_metadata[ch] = {
                                            "volume_mL": match["volume_mL"].iloc[0],
                                            "coral_ID": match["coral_ID"].iloc[0],
                                           }
                    if debug:
                        st.write(f"✅ Match for {filename} | {ch}")
    
                else:
                    channel_metadata[ch] = {
                                            "volume_mL": match["volume_mL"].iloc[0],
                                            "coral_ID": match["coral_ID"].iloc[0],
                                           }
                    if debug:
                        st.write(f"⚠️ Multiple matches for {filename} | {ch} ({len(match)} rows). Using first.")
    
            edited_raw_dfs_dict[clean_filename]["channel_metadata"] = channel_metadata
    
            if debug:
                st.write(channel_metadata)

    
        return edited_raw_dfs_dict
            
    
    
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
    
    def agtable(self, 
                all_regression_results, 
                base_row_height: int = 30, 
                max_rows: int = 20, 
                use_checkboxes: bool = True,
                show_warning: bool = True):
        
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
                    
                maxlag = results.get("MaxLag")
                slope = results.get("slope (umol/L/s)")
                slope_ci = results.get("slope 95% CI (umol/L/s)")
                R2 = results.get("R2")
                SSR = results.get("squared residuals")
                rmse = results.get("RMSE")
                pval = results.get("slope pval")
                stderr = results.get("slope stderr (umol/L/s)")
                coral_id = results.get("Coral ID")
                vol = results.get("Volume (mL)")
                noise_percent = results.get("Noise %")
                rse_percent = results.get("slope % RSE")
                zmax = results.get("Zmax")
 
                # Round numeric values
                slope = round(slope, 4) if isinstance(slope, (float,int)) else np.nan
                # slope_ci = round(slope_ci, 4) if isinstance(slope_ci, (float,int)) else np.nan
                R2 = round(R2, 3) if isinstance(R2, (float,int)) else np.nan
                SSR = round(SSR, 2) if isinstance(SSR, (float,int)) else np.nan
                rmse = round(rmse, 3) if isinstance(rmse, (float, int)) else np.nan
                noise_percent = round(noise_percent, 1) if isinstance(noise_percent, (float, int)) else np.nan
                maxlag = maxlag if isinstance(maxlag, (float,int)) else np.nan
                rse_percent = round(rse_percent, 2) if isinstance(rse_percent, (float,int)) else np.nan
                zmax = round(zmax, 2) if isinstance(zmax, (float,int)) else np.nan
    
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
                        "slope (umol/L/s)": slope,
                        "R2": R2,
                        "slope 95% CI (umol/L/s)": slope_ci,
                        "slope % RSE":rse_percent,
                        # "slope pval": pval_str,
                        # "p-value_numeric": pval if isinstance(pval,(float,int)) else np.nan,
                        # "slope stderr (umol/L/s)": stderr_str,
                        # "SSR": SSR,
                        # "RMSE":rmse,
                        "Noise %":noise_percent,
                        "Zmax":zmax,
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
        # Maxlags and SSR are commented out with //
        cell_style_jscode = JsCode("""
                                    function(params) {
                                        if(params.value != null && params.colDef.field == 'R2' && params.value < 0.90) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'Noise %' && params.value > 25) {
                                        return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'slope % RSE' && params.value > 50.0) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        if(params.value != null && params.colDef.field == 'Zmax' && params.value > 4.0) {
                                            return {'backgroundColor':'yellow', 'color':'black'};
                                        }
                                        
                                        //if(params.value != null && params.colDef.field == 'SSR' && params.value > 1000) {
                                            //return {'backgroundColor':'yellow', 'color':'black'};
                                        //}
                                        //if(params.value != null && params.colDef.field == 'MaxLag' && params.value > 20) {
                                            //return {'backgroundColor':'yellow', 'color':'black'};
                                        //}
                                        //if(params.data['p-value_numeric'] > 0.05) {
                                           // return {'backgroundColor':'yellow', 'color':'black'};
                                        //}
                                        return null;
                                    };
                                    """)
    
        # Apply JS styling to each column that needs it
        # highlight_columns = ["slope pval", "R2", "SSR", "MaxLag"]
        highlight_columns = ["R2", "Noise %", "slope % RSE", "Zmax"]
        
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
        # Controlled by session_state['warning'] and ['warning_rendered'] so that it only runs once
        # per upload and only after the first aggrid table
        if not st.session_state["warning"] and (df["MaxLag"] > 20).any():
   
            col1, col2, col3 = st.columns([1, 3, 1])

            if show_warning:
                
                # Use a placeholder to control exact layout
                placeholder = col2.empty()  
                
                with placeholder.container():
                    st.error(
                            "Warning! MaxLag is high for the channels highlighted.\n"
                            "This indicates high autocorrelation in the data that yields low estimates\n"
                            "for R2 and slope standard error.\n"
                            "This has been corrected for using the Newey-West (NW) correction in the Model Diagnostics tab."
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
                                                "metadata": {channel: {"coral_id": coral_id, "volume": volume}}
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
                        subset_data[filename]["metadata"][channel] = {"coral_id": coral_id, "volume": volume} 
                       
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

             # Add legend to the right-hand side
            ax.legend(
                      loc="center left",
                      bbox_to_anchor=(1, 0.5),
                      frameon=False
                     )
            
            ax.grid(True)
            plt.tight_layout()
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
            slope_ci = f"{ci_low:.4g}–{ci_high:.4g}"
        
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
                            "slope (umol/L/s)":slope,
                            "intercept":intercept,
                            "R2":R2,
                            "slope pval":p,
                            "slope stderr (umol/L/s)":stderr,
                            "CI Low (umol/L/s)":ci_low,
                            "CI High (umol/L/s)":ci_high,
                            "slope 95% CI (umol/L/s)":slope_ci,
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
                nw_slope = nw_model.params[1]
                se_slope = nw_model.bse[1]
                pval_slope = "{:.3e}".format(nw_model.pvalues[1])
                r2 = nw_model.rsquared
                
                ci_low = nw_slope - z * se_slope
                ci_high = nw_slope + z * se_slope
                
                # nw_slope_ci = f"'{ci_low:.6f} - {ci_high:.6f}"
                nw_slope_ci = f"{ci_low:.4g}–{ci_high:.4g}"
                
                results_dict[fname][ch] = {
                                            # "intercept": intercept,
                                            "NW slope (umol/L/s)": nw_slope,
                                            "NW slope stderr (umol/L/s)": se_slope,
                                            "NW slope pval": pval_slope,
                                            "NW slope 95% CI (umol/L/s)": nw_slope_ci,
                                            "MaxLag": lags
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
                    "slope (umol/L/s)",
                    "R2",
                    "slope stderr (umol/L/s)",
                    "slope pval",
                    "slope 95% CI (umol/L/s)",
                    "NW slope (umol/L/s)",
                    "NW slope stderr (umol/L/s)",
                    "NW slope pval",
                    "NW slope 95% CI (umol/L/s)",
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
                    if key in ["slope", "NW slope (umol/L/s)"]:
                        val = round(val, 4)
                    elif key == "R2":
                        val = round(val, 3)
                    elif key in ["slope stderr (umol/L/s)", "NW slope stderr (umol/L/s)"]:
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
    
                # step = max(1, int(window_size * step_perc * 0.1))
                step = max(1, int(window_size * (step_perc / 100.0)))
                
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
                    nw_slope_rnd = round(nw_slope, 4) if isinstance(nw_slope, (float,int)) else np.nan
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
                                    "Slope (umol/L/s)": round(slope, 4),
                                    "intercept": round(intercept, 4),
                                    "Slope Stderr (umol/L/s)": round(stderr, 6),
                                    "R2": round(r2, 4),
                                    "Slope pval": f"{nw_model.pvalues[1]:.2e}",
                                    "Slope 95% CI (umol/L/s)": slope_ci,
                                    "SSR": round(ssr, 3),
                                    "NW Slope (umol/L/s)":nw_slope_rnd,
                                    "NW Slope 95% CI (umol/L/s)": f"{ci_low_nw:.6f} - {ci_high_nw:.6f}",
                                    "NW pval": f"{nw_pval:.2e}",
                                    "NW Slope Stderr (umol/L/s)": round(nw_se, 6),
                                    "MaxLag": lags
                                  }
    
                    heapq.heappush(heap, (r2, window_dict))
                    if len(heap) > 200:
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
                       top_n_display: int = 5,
                       base_row_height: int = 30, 
                       max_rows: int = 20):
        
        if results_df.empty:
            st.warning("Not enough data for rolling regression.")
            return
        
        # Only show top n results by R2
        df_show = results_df.copy()

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
        
        # Select top N per Filename+Channel by R² (SSR tie-break if present)
        sort_cols = ["R2"]
        ascending = [False]
        if "SSR" in df_show.columns:
            sort_cols.append("SSR")
            ascending.append(True)
        
        # Sort by R2 first and then SSR for ties
        # Delete all but the number denoted by ton_n_display
        df_show = (
                   df_show.sort_values(sort_cols, ascending=ascending)
                       .groupby(["Filename", "Channel"], sort=False, as_index=False)
                       .head(top_n_display)
                  )
        
        # Final display order:
        # Filename (original order) -> channel (Ch1..Ch4) -> R² best-to-worst
        df_show = (
                   df_show.sort_values(["_fname_order", "_ch_order", "R2"],
                                    ascending=[True, True, False])
                       .drop(columns=["_fname_order", "_ch_order"])
                       .reset_index(drop=True)
                  )
    
        # If you don't want to show all the columns in the df, you can
        # put some in a side bar.
        
        # Define columns to hide
        hidden_cols = ["SSR", 
                       "intercept", 
                       "MaxLag", 
                       "Window", 
                       # "Start", 
                       # "End", 
                       "NW Slope (umol/L/s)",
                       "NW Slope Stderr (umol/L/s)",
                       "NW pval",
                       "NW Slope 95% CI (umol/L/s)", 
                       ]

        # Build AgGrid options
        gb = GridOptionsBuilder.from_dataframe(df_show)
        
        # Apply hide=True to any columns in the hidden list
        # Hide selected columns
        for col in hidden_cols:
            gb.configure_column(col, hide=True)
        
        # Enable column selection sidebar
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_side_bar(columns_panel=True)
        
        first = df_show.columns[0]
        second = df_show.columns[1]
        
        gb.configure_column(first, width=220, minWidth=180)
        gb.configure_column(second, width=140, minWidth=120)
        
        grid_options = gb.build()
       
        # also good as a general safety net:
        gb.configure_default_column(resizable=True, minWidth=110)
        
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
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                height=height,
                                # fit_columns_on_grid_load=True,
                                allow_unsafe_jscode=True,
                                enable_enterprise_modules=True,
                                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
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
            slope, intercept = selected["Slope (umol/L/s)"][0], selected["intercept"][0]

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
                            showlegend = False,
                            width=900,
                            height=400 + 100 * ((len(final_x_y_valid_dict[sel_filename]) - 1) // 2),
                            margin=dict(l=80, r=40, t=60, b=60)
                           )

            st.plotly_chart(fig, use_container_width=True)
            
    def download_regression(self, all_subset_regs):
        """Create a download button for regressions of the data after the user
        adjusts the x-axis."""
        
        # st.write(all_subset_regs)
      
        rows = []
        
        for filename, channels in all_subset_regs.items():     
            for ch, results in channels.items():
              
                slope_CI = results.get("slope 95% CI (umol/L/s)")
                rows.append({
                            "Filename": filename,
                            "Channel": ch,
                            "Coral ID": results["Coral ID"],
                            "Volume (mL)": results["Volume (mL)"],
                            "Start Time": results.get("Start Time", np.nan),
                            "Stop Time": results.get("Stop Time", np.nan),
                            "slope (umol/L/s)": results.get("slope (umol/L/s)", np.nan),
                            "R2": round(results["R2"], 2) if isinstance(results.get("R2"), (float, int)) else np.nan,
                            "slope 95% CI (umol/L/s)": slope_CI,
                            "slope % RSE":round(results["slope % RSE"], 2) if isinstance(results.get("slope % RSE"), (float, int)) else np.nan,
                            "Noise %": round(results["Noise %"], 1) if isinstance(results.get("Noise %"), (float, int)) else np.nan,
                            "Zmax":round(results["Zmax"], 1) if isinstance(results.get("Zmax"), (float, int)) else np.nan,
                            "slope pval": results.get("slope pval", np.nan),
                            "slope stderr (umol/L/s)": results.get("slope stderr (umol/L/s)", np.nan),
                            "SSR": results.get("squared residuals", np.nan),
                            "MaxLag": results.get("MaxLag", np.nan),
                           })
    
        # Raw dataframe (for CSV)
        df_raw = pd.DataFrame(rows)
    
        # Formatted copy (rounded for display only)
        df_display = df_raw.copy()
        
        df_display["slope (umol/L/s)"] = df_display["slope (umol/L/s)"].apply(
            lambda x: round(x, 4) if isinstance(x, (float, int)) else x
        )
        df_display["R2"] = df_display["R2"].apply(
            lambda x: round(x, 3) if isinstance(x, (float, int)) else x
        )
        
        df_display["slope % RSE"] = df_display["slope % RSE"].apply(
            lambda x: round(x,2) if isinstance(x, (float, int)) else x
        )
        
        # Other options
        df_display["SSR"] = df_display["SSR"].apply(
            lambda x: round(x, 2) if isinstance(x, (float, int)) else x
        )
        df_display["slope pval"] = df_display["slope pval"].apply(
            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) else x
        )
        df_display["slope stderr (umol/L/s)"] = df_display["slope stderr (umol/L/s)"].apply(
            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) else x
        )
    
        # Show formatted table
        st.markdown(
                    "<p style='color: Blue; font-size: 24px; margin: 0;'>All Adjusted Regression Results</p>",
                    unsafe_allow_html=True
                   )
        
        st.data_editor(df_display, use_container_width=True, num_rows="static", hide_index=True)
    
        # Download raw values
        csv = df_raw.to_csv(index=False).encode("utf-8-sig")
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
    # @st.cache_data(show_spinner=False)
    def kde(self, 
            rolling_df: pd.DataFrame,
            slope_col: str = "Slope (umol/L/s)",
            title: str = "Most Consistent Respiration Rate",
            button_text: str = "Download Most Consistent Rates",
            file_name: str = "kde_summary.csv",
            # KDE + interval controls
            grid_n: int = 400,
            tol_sd_mult: float = 1.0,
            min_slopes: int = 10,
            min_r2: float | None = None,   
            # display controls
            sort_by: str = "Stable Duration (s)", 
           ) -> pd.DataFrame:
        
        """
        Workflow:
          Step 1: Compute KDE-based summary (one row per Filename x Channel)
          Step 2: Display the table in Streamlit
          Step 3: Provide a download button
    
        Expects rolling_df to include: Filename, Channel, Start, End, R2, and slope_col.
        Returns the summary dataframe (also displayed).
        """
      
        if rolling_df is None or getattr(rolling_df, "empty", True):
            st.warning("KDE summary: rolling_df is empty.")
            return pd.DataFrame()
    
        df = rolling_df.copy()
       
        # Numeric coercion
        df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
        df["End"] = pd.to_numeric(df["End"], errors="coerce")
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    
        # Check for R2
        if min_r2 is not None:
            df = df[df["R2"].notna() & (df["R2"] >= float(min_r2))]
    
        rows = []
    
        # Calculate KDE per (Filename, Channel) 
        for (fname, ch), g in df.groupby(["Filename", "Channel"], sort=False):
            slopes = g[slope_col].dropna().to_numpy(dtype=float)
            if slopes.size < int(min_slopes):
                continue
    
            sd = float(np.std(slopes))
            tol = float(tol_sd_mult * sd) if sd > 0 else 1e-12
    
            # KDE mode (fallback to median if variance ~0)
            if not np.isfinite(sd) or sd < 1e-12:
                mode = float(np.median(slopes))
            else:
                kde_obj = gaussian_kde(slopes)
                grid = np.linspace(float(np.min(slopes)), float(np.max(slopes)), int(grid_n))
                dens = kde_obj(grid)
                mode = float(grid[int(np.argmax(dens))])
    
            # "near mode" windows -> interval estimate
            near = g[np.abs(g[slope_col] - mode) <= tol].copy()
            if near.empty:
                best = g.sort_values(["R2"], ascending=[False]).iloc[0]
                stable_start = float(best["Start"])
                stable_end = float(best["End"])
                near_frac = 0.0
            else:
                stable_start = float(near["Start"].min())
                stable_end = float(near["End"].max())
                near_frac = float(len(near) / len(g)) if len(g) else 0.0
    
            best_row = g.sort_values(["R2"], ascending=[False]).iloc[0]
            best_r2 = float(best_row["R2"])
            best_r2_slope = float(best_row[slope_col])
    
            rows.append({
                        "Filename": fname,
                        "Channel": ch,
                        # "Stable Start (s)": stable_start,
                        # "Stable End (s)": stable_end,
                        "Most Common Respiration Rate (umol/L)": mode,
                        "Rate Stability (STDEV)": sd,
                        # "Rate with Highest R²": best_r2_slope,
                        # "Best R²": best_r2,
                        # "Mode − BestR² (abs)": abs(mode - best_r2_slope),
                        # "Stable Duration (s)": stable_end - stable_start,
                        # "Frac windows near mode": near_frac,
                        "Number of windows wide": int(len(g)),
                       })
      
        summary = pd.DataFrame(rows)
    
        st.write('')
        
        # Display 
        st.markdown(
                    f"<p style='color: dodgerblue; font-size: 20px; margin: 0;'><b>{title}</b></p>",
                    unsafe_allow_html=True
                   )
        st.write("")
    
        if summary.empty:
            st.warning("No KDE summary available (not enough rolling windows per channel after filtering).")
            return summary
    
        # Rounding for display
        summary = summary.copy()
        summary["Most Common Respiration Rate (umol/L)"] = summary["Most Common Respiration Rate (umol/L)"].round(4)
        summary["Rate Stability (STDEV)"] = summary["Rate Stability (STDEV)"].round(4)
        # summary["Rate R²"] = summary["Rate R²"].round(4)
        # summary["Best R²"] = summary["Best R²"].round(4)
        # summary["Mode − BestR² (abs)"] = summary["Mode − BestR² (abs)"].round(4)
        # summary["Stable Start (s)"] = summary["Stable Start (s)"].round(1)
        # summary["Stable End (s)"] = summary["Stable End (s)"].round(1)
        # summary["Stable Duration (s)"] = summary["Stable Duration (s)"].round(1)
        # summary["Frac windows near mode"] = summary["Frac windows near mode"].round(3)
    
        # sort
        if sort_by in summary.columns:
            summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True) 
            
        summary_str = summary.astype(str)
    
        st.dataframe(summary_str, use_container_width=False, hide_index=True)
        
        # Download
        csv = summary.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
                            label=button_text,
                            data=csv,
                            file_name=file_name,
                            mime="text/csv",
                          )
    
        return summary
                    
                
            
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
    
    # Get the path relative to the current file (inside Docker container)
    BASE_DIR = os.path.dirname(__file__)
        
    # Use this for cloud
    st.session_state['logo'] = 'mote_logo.png'
    
    # Load image for favicon
    logo_img = Image.open(st.session_state['logo'])
    
    # Use this for local machine
    # if 'logo' not in st.session_state:
    #     st.session_state['logo'] = BASE_DIR + '/mote_logo.png'
        
    # logo_img = BASE_DIR + '/mote_logo.png'
    
        
    # Page config
    st.set_page_config(layout = "wide", 
                       page_title = 'Mote', 
                       page_icon = logo_img,
                       initial_sidebar_state="auto", 
                       menu_items = None)
    
    
    # Call Flow_Control class that makes all calls to other classes and methods
    obj1 = Flow_Control()
    all_calls = obj1.all_calls()
