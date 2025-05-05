# Data-Driven-Stock-Analysis

This project provides a comprehensive stock analysis for Nifty 50 companies using data from December 2023 to November 2024. It includes a Python-based Streamlit dashboard and interactive Power BI visualizations, offering rich insights into market trends, volatility, sector performance, price correlation, and monthly stock gainers/losers.

Project Overview:
Data Source: Monthly YAML files for 50 Nifty stocks.
Data Conversion: yaml_to_csv.py script converts raw YAML files in stocks_dataset/ into clean CSV files stored in symbol_data/.
Preprocessing & Analysis: Conducted in stock_file.ipynb, which loads and prepares data for advanced analytics and dashboard integration.

Visualization:
Streamlit App (main.py): Matplotlib/Seaborn Visualization for Python-based analytics.
Power BI Visuals (Data Driven Stock Analysis.pbix): Interactive dashboard using the consolidated dataset (combined_stock_data.csv).

Streamlit and Power BI common Dashboard Features
The dashboard provides:
Top 10 Green Stocks – Based on yearly return.
Top 10 Loss Stocks – Sorted by worst yearly performance.
Market Summary – Number of green (positive return) vs. red (negative return) stocks.
Volatility Analysis – Standard deviation of daily returns to understand stock risk.
Cumulative Return Over Time – Performance trend over the year for each stock.
Sector-wise Performance – Insightful view using sector mapping (CSV based).
Stock Price Correlation – Heatmap showing how stocks move together.
Monthly Gainers/Losers – Top 5 monthly winners and losers per month.

Setting Up the Project:
Recommended to use a virtual environment before creating any files to avoid dependency conflicts. use this command,
python -m venv your_virtual_environment_folder_name

We have to activate scripts to use the newly created virtual environment. To activate scripts,
For Windows:
.\your_virtual_environment_folder_name\Scripts\activate
For macOS/Linux:
source your_virtual_environment_folder_name/bin/activate

Install Required Packages. Use the following command to install all necessary packages:
pip install streamlit pandas matplotlib seaborn sqlalchemy pyyaml

Packages used across the project:
streamlit – For interactive web dashboard.
pandas, numpy – For data handling.
matplotlib, seaborn – For data visualization.
sqlalchemy – For database integration.
pyyaml – For parsing YAML files.
datetime, os, re, csv, collections – Core libraries for file and data handling.

Run the app via:
streamlit run main.py
