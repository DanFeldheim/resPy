#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:39:40 2024

CSS styles file.

@author: danfeldheim
"""

CSS = """
        <style>
        /* Font colors */
        .blue-36 {
            font-size: 36px !important;
            color: blue;
        }
        .blue-30 {
            font-size: 30px !important;
            color: blue;
        }
        .purple-24 {
            font-size: 24px !important;
            color: purple;
        }
        .blueviolet-28 {
            font-size: 28px !important;
            color: blueviolet;
        }
        .green-18 {
            font-size: 18px !important;
            color: green;
        }
        .green-24 {
            font-size: 24px !important;
            color: darkgreen;
            font-weight: bold;
        }
        
        .DarkBlue-24 {
            font-size:24px !important;
            color:DarkBlue;
            font-weight: bold;}
                        
        
        /* Global button style */
        div.stButton > button:first-child {
            background-color: rgb(0, 102, 102);
            color: white;
            font-size: 20px;
            font-weight: bold;
            display: block;
        }
        div.stButton {
            display: flex;
            justify-content: flex-start;
        }
        div.stButton > button:hover {
            background: linear-gradient(to bottom, rgb(0, 204, 204) 5%, rgb(0, 204, 204) 100%);
            background-color: rgb(0, 204, 204);
        }
        div.stButton > button:active {
            position: relative;
            top: 3px;
        }
        
        /* Sidebar color */
        [data-testid=stSidebar] {
            background-color: LightSteelBlue;
        }
        
        /* Top padding of app */
        .block-container {
            padding-top: 0rem;
        }
        
        /* Widget text styling */
        div[class*="stTextArea"] label p {
            font-size: 18px;
            color: DarkBlue;
        }
        div[class*="stTextInput"] label p {
            font-size: 18px;
            color: DarkBlue;
        }
        div[class*="stNumberInput"] label p {
            font-size: 18px;
            color: green;
        }
        div[class*="stDateInput"] label p {
            font-size: 18px;
            color: DarkBlue;
        }
        div[class*="stFileUploader"] label p {
            font-size: 18px;
            color: green;
        }
        div[data-testid="stSelectbox"] label {
           font-size: 18px; 
           color: DarkBlue;    
       }
        </style>
        """