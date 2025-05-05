# Use this code to convert 12 months of stock data yaml file to 50 csv files for each symbol (50 symbols)

import yaml
import csv
import os
import re
from collections import defaultdict
import pandas as pd
from datetime import datetime

# Configuration
BASE_DIR = r'your main folder path'  # ← CHANGE THIS TO YOUR MAIN FOLDER PATH
# Example: BASE_DIR = r'C:\Users\YourName\Documents\Stocks'
STOCKS_DIR = os.path.join(BASE_DIR, 'stocks_dataset')
CSV_DIR = os.path.join(BASE_DIR, 'csv_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'symbol_data')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load sector mapping from CSV
try:
    sectors_df = pd.read_csv(os.path.join(CSV_DIR, 'sectors.csv'))
    symbol_to_sector = {
        symbol.split(':')[1].strip() if ':' in symbol else symbol.strip(): sector 
        for symbol, sector in zip(sectors_df['Symbol'], sectors_df['sector'])
    }
    print("✓ Loaded sector mapping")
except Exception as e:
    print(f"⚠ Sector error: {str(e)}")
    symbol_to_sector = {}

# Process YAML files
symbol_data = defaultdict(list)
existing_records = set()

for month_folder in sorted(os.listdir(STOCKS_DIR)):
    month_path = os.path.join(STOCKS_DIR, month_folder)
    if not os.path.isdir(month_path):
        continue
    
    print(f"Processing {month_folder}...")
    
    for yaml_file in sorted(os.listdir(month_path)):
        if not yaml_file.lower().endswith(('.yaml', '.yml')):
            continue
        
        try:
            with open(os.path.join(month_path, yaml_file), 'r') as f:
                data = yaml.safe_load(f) or []
                
                for stock in data:
                    symbol = stock.get('Ticker', '').strip()
                    if not symbol:
                        continue
                    
                    # Skip duplicates
                    record_id = f"{symbol}|{stock.get('date','').split()[0]}"
                    if record_id in existing_records:
                        continue
                    
                    # Add new 'symbol' column (MIRRORS 'Ticker')
                    stock['symbol'] = symbol  # ← NEW COLUMN ADDED
                    
                    # Add sector
                    stock['sector'] = symbol_to_sector.get(symbol, 'UNKNOWN')
                    
                    symbol_data[symbol].append(stock)
                    existing_records.add(record_id)
                    
        except Exception as e:
            print(f"Error in {yaml_file}: {str(e)}")

# Write CSVs with new 'symbol' column
for symbol, data in symbol_data.items():
    if not data:
        continue
        
    clean_symbol = re.sub(r'[^a-zA-Z0-9]', '_', symbol)
    output_file = os.path.join(OUTPUT_DIR, f"{clean_symbol}.csv")
    
    df = pd.DataFrame(data)

    try:
        # Convert date to datetime.date
        stock['date'] = pd.to_datetime(stock['date'],format='%Y-%m-%d %H:%M:%S')
        
    except Exception as e:
        print(f"Date conversion error for {symbol}: {str(e)}")
        continue

    # Ensure 'symbol' is included in fieldnames
    fieldnames = sorted({'symbol', 'date'}.union(  # ← GUARANTEES 'symbol' COLUMN
        {field for entry in data for field in entry.keys()}
    ))
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {clean_symbol}.csv ({len(data)} records)")

print("✅ Done!")