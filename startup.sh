#!/bin/bash

# Ensure Python dependencies are installed
pip install --no-cache-dir -r requirements.txt

# Run Streamlit app in detached mode on port 7860
streamlit run --server.port 7860 app.py
