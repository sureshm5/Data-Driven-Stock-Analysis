import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from datetime import datetime

# --- Database Connection Details ---
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'your_password'  # Replace with your actual password
DB_PORT = 3306
DB_NAME = 'your_database_name'  # Replace with your actual database name

DATABASE_URL = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

def get_table_names():
    """Fetch all table names from the database"""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            all_tables_query = text("SHOW TABLES")
            result = connection.execute(all_tables_query)
            table_names = [row[0] for row in result]
            return table_names
    except Exception as e:
        st.error(f"Error fetching table names from the database: {e}")
        return None

def task0_market_summary(engine):
    """Display key metrics and market summary using SQL queries"""
    st.subheader("Market Summary and Key Metrics")
    
    # Get first and last prices for each stock
    yearly_returns = {}
    table_names = get_table_names()
    
    if not table_names:
        st.error("No tables found in the database.")
        return
    
    for table in table_names:
        try:
            with engine.connect() as connection:
                # Get first and last close prices with dates
                query = text(f"""
                    SELECT 
                        symbol, 
                        (SELECT close FROM `{table}` ORDER BY date ASC LIMIT 1) as first_price,
                        (SELECT close FROM `{table}` ORDER BY date DESC LIMIT 1) as last_price
                    FROM `{table}` LIMIT 1
                """)
                result = connection.execute(query)
                row = result.fetchone()
                
                if row and row[1] and row[2]:
                    symbol = row[0]
                    first_price = float(row[1])
                    last_price = float(row[2])
                    yearly_return = ((last_price - first_price) / first_price) * 100
                    yearly_returns[symbol] = yearly_return
                    
        except Exception as e:
            st.warning(f"Error processing table {table}: {e}")
            continue

    if yearly_returns:
        sorted_returns = sorted(yearly_returns.items(), key=lambda item: item[1], reverse=True)
        top_10_green = sorted_returns[:10]
        top_10_red = sorted(yearly_returns.items(), key=lambda item: item[1])[:10]

        st.subheader("Top 10 Green Stocks (Yearly Return)")
        green_df = pd.DataFrame(top_10_green, columns=['Symbol', 'Yearly Return (%)'])
        green_df.index = green_df.index+1
        st.table(green_df)
        
        fig_green, ax_green = plt.subplots()
        ax_green.bar([item[0] for item in top_10_green], [item[1] for item in top_10_green], color='green')
        ax_green.set_xlabel('Stock Symbol')
        ax_green.set_ylabel('Yearly Return (%)')
        ax_green.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_green)

        st.subheader("Top 10 Red Stocks (Yearly Return)")
        red_df = pd.DataFrame(top_10_red, columns=['Symbol', 'Yearly Return (%)'])
        red_df.index = red_df.index+1
        st.table(red_df)
        
        fig_red, ax_red = plt.subplots()
        ax_red.bar([item[0] for item in top_10_red], [item[1] for item in top_10_red], color='red')
        ax_red.set_xlabel('Stock Symbol')
        ax_red.set_ylabel('Yearly Return (%)')
        ax_red.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_red)

        green_count = sum(1 for ret in yearly_returns.values() if ret > 0)
        red_count = sum(1 for ret in yearly_returns.values() if ret < 0)
        total_stocks = len(yearly_returns)
        green_percentage = (green_count / total_stocks) * 100 if total_stocks > 0 else 0
        red_percentage = (red_count / total_stocks) * 100 if total_stocks > 0 else 0

        st.subheader("Market Overview")
        overview_data = pd.DataFrame({
            "Metric": ["Number of Green Stocks", "Number of Red Stocks", "Percentage of Green Stocks", "Percentage of Red Stocks"],
            "Value": [green_count, red_count, f"{green_percentage:.2f}%", f"{red_percentage:.2f}%"]
        })
        overview_data.index = overview_data.index+1
        st.table(overview_data)

        fig_pie, ax_pie = plt.subplots()
        labels = ['Green Stocks', 'Red Stocks']
        sizes = [green_percentage, red_percentage]
        colors = ['lightgreen', 'lightcoral']
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        ax_pie.axis('equal')
        plt.title('Market Overview: Percentage of Green vs. Red Stocks')
        st.pyplot(fig_pie)
    else:
        st.warning("Could not calculate yearly returns.")

def task1_volatility_analysis(engine):
    """Display volatility analysis using SQL queries"""
    st.subheader("Volatility Analysis")
    
    stock_volatility = {}
    table_names = get_table_names()
    
    if not table_names:
        st.error("No tables found in the database.")
        return
    
    for table in table_names:
        try:
            with engine.connect() as connection:
                # Calculate daily returns and standard deviation (volatility) in SQL
                query = text(f"""
                    WITH daily_returns AS (
                        SELECT 
                            symbol,
                            (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) AS daily_return
                        FROM `{table}`
                    )
                    SELECT 
                        symbol,
                        STDDEV(daily_return) AS volatility
                    FROM daily_returns
                    WHERE daily_return IS NOT NULL
                    GROUP BY symbol
                """)
                result = connection.execute(query)
                row = result.fetchone()
                
                if row and row[1]:
                    symbol = row[0]
                    volatility = float(row[1])
                    stock_volatility[symbol] = volatility
                    
        except Exception as e:
            st.warning(f"Error processing table {table}: {e}")
            continue

    if stock_volatility:
        sorted_volatility = sorted(stock_volatility.items(), key=lambda item: item[1], reverse=True)
        top_10_volatile = sorted_volatility[:10]
        volatile_df = pd.DataFrame(top_10_volatile, columns=['Symbol', 'Volatility'])
        st.subheader("Top 10 Most Volatile Stocks (Past Year)")
        volatile_df.index = volatile_df.index+1
        st.table(volatile_df)
        
        fig_volatility, ax_volatility = plt.subplots()
        ax_volatility.bar([item[0] for item in top_10_volatile], [item[1] for item in top_10_volatile], color='skyblue')
        ax_volatility.set_xlabel('Stock Symbol')
        ax_volatility.set_ylabel('Volatility (Standard Deviation of Daily Returns)')
        ax_volatility.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_volatility)
    else:
        st.warning("Could not calculate stock volatility.")

def task2_cumulative_returns(engine):
    """Display cumulative return analysis using SQL queries"""
    st.subheader("Cumulative Return Over Time (Top 5)")
    
    try:
        with engine.connect() as connection:
            # First create the temporary table
            connection.execute(text("""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_cumulative_returns (
                    symbol VARCHAR(50),
                    cumulative_return DECIMAL(20, 10),
                    PRIMARY KEY (symbol)
                )
            """))
            
            # Clear any existing data
            connection.execute(text("TRUNCATE TABLE temp_cumulative_returns"))
            
            # Get all table names
            table_names = get_table_names()
            
            # Calculate and insert cumulative returns for each stock
            for table in table_names:
                try:
                    # Calculate cumulative returns in memory first
                    query = text(f"""
                        WITH daily_returns AS (
                            SELECT 
                                symbol,
                                date,
                                (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) AS daily_return
                            FROM `{table}`
                        ),
                        cumulative_returns AS (
                            SELECT 
                                symbol,
                                date,
                                EXP(SUM(LN(1 + daily_return)) OVER (ORDER BY date)) - 1 AS cumulative_return
                            FROM daily_returns
                            WHERE daily_return IS NOT NULL
                        )
                        SELECT 
                            symbol,
                            MAX(cumulative_return) as cumulative_return
                        FROM cumulative_returns
                        GROUP BY symbol
                    """)
                    
                    # Fetch the results
                    result = connection.execute(query)
                    for row in result:
                        # Insert into temp table
                        connection.execute(
                            text("""
                                INSERT INTO temp_cumulative_returns (symbol, cumulative_return)
                                VALUES (:symbol, :cumulative_return)
                                ON DUPLICATE KEY UPDATE cumulative_return = VALUES(cumulative_return)
                            """),
                            {'symbol': row[0], 'cumulative_return': float(row[1])}
                        )
                        
                except Exception as e:
                    st.warning(f"Error processing table {table}: {e}")
                    continue
            
            # Get top 5 performing stocks
            query = text("""
                SELECT symbol, cumulative_return 
                FROM temp_cumulative_returns 
                ORDER BY cumulative_return DESC 
                LIMIT 5
            """)
            top_5_performing = [(row[0], float(row[1])) for row in connection.execute(query)]
            
            if not top_5_performing:
                st.warning("Could not calculate cumulative returns.")
                return
            
            # Display the top performers table
            st.subheader("Cumulative Return Over Time for Top 5 Performing Stocks")
            top_5_df = pd.DataFrame(top_5_performing, columns=['Symbol', 'Final Cumulative Return'])
            top_5_df['Final Cumulative Return'] = top_5_df['Final Cumulative Return'].map('{:.2%}'.format)
            top_5_df.index = top_5_df.index + 1
            st.table(top_5_df)
            
            # Plot the cumulative returns over time for top 5
            fig_cumulative, ax_cumulative = plt.subplots(figsize=(10, 6))
            
            for symbol, _ in top_5_performing:
                try:
                    query = text(f"""
                        WITH daily_returns AS (
                            SELECT 
                                date,
                                (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) AS daily_return
                            FROM `{symbol}`
                        ),
                        cumulative_returns AS (
                            SELECT 
                                date,
                                EXP(SUM(LN(1 + daily_return)) OVER (ORDER BY date)) - 1 AS cumulative_return
                            FROM daily_returns
                            WHERE daily_return IS NOT NULL
                        )
                        SELECT date, cumulative_return
                        FROM cumulative_returns
                        ORDER BY date
                    """)
                    df = pd.read_sql(query, connection)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        ax_cumulative.plot(df['date'], df['cumulative_return'], label=symbol)
                        
                except Exception as e:
                    st.warning(f"Error processing stock {symbol}: {e}")
                    continue

            ax_cumulative.set_xlabel('Date')
            ax_cumulative.set_ylabel('Cumulative Return')
            ax_cumulative.set_title('Cumulative Return Over Time for Top 5 Performing Stocks')
            ax_cumulative.legend(loc='upper left')
            ax_cumulative.grid(True)
            plt.tight_layout()
            st.pyplot(fig_cumulative)
            
    except Exception as e:
        st.error(f"Error calculating cumulative returns: {e}")

    

def task3_sector_performance(engine):
    """Display sector-wise performance analysis using SQL queries"""
    st.subheader("Sector-wise Performance")
    
    try:
        with engine.connect() as connection:
            # Create temporary table for sector returns
            connection.execute(text("""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_sector_data (
                    sector VARCHAR(100) PRIMARY KEY,
                    total_return DECIMAL(20, 4),
                    stock_count INT
                )
            """))
            
            # Clear any existing data
            connection.execute(text("TRUNCATE TABLE temp_sector_data"))
            
            # Get all table names
            table_names = get_table_names()
            
            if not table_names:
                st.error("No tables found in the database.")
                return
            
            # Calculate yearly return for each stock and aggregate by sector
            for table in table_names:
                try:
                    # Get sector and yearly return for this stock
                    query = text(f"""
                        WITH first_last AS (
                            SELECT 
                                sector,
                                (SELECT close FROM `{table}` ORDER BY date ASC LIMIT 1) as first_price,
                                (SELECT close FROM `{table}` ORDER BY date DESC LIMIT 1) as last_price
                            FROM `{table}` 
                            LIMIT 1
                        )
                        SELECT 
                            sector,
                            (last_price - first_price) / first_price * 100 as yearly_return
                        FROM first_last
                        WHERE sector IS NOT NULL 
                          AND first_price IS NOT NULL 
                          AND last_price IS NOT NULL
                    """)
                    
                    result = connection.execute(query)
                    row = result.fetchone()
                    
                    if row:
                        sector = row[0]
                        yearly_return = float(row[1])
                        
                        # Update sector aggregates
                        connection.execute(
                            text("""
                                INSERT INTO temp_sector_data (sector, total_return, stock_count)
                                VALUES (:sector, :yearly_return, 1)
                                ON DUPLICATE KEY UPDATE 
                                    total_return = total_return + VALUES(total_return),
                                    stock_count = stock_count + 1
                            """),
                            {'sector': sector, 'yearly_return': yearly_return}
                        )
                        
                except Exception as e:
                    st.warning(f"Error processing table {table}: {e}")
                    continue
            
            # Calculate average returns by sector
            query = text("""
                SELECT 
                    sector,
                    total_return/stock_count as avg_return,
                    stock_count
                FROM temp_sector_data
                ORDER BY avg_return DESC
            """)
            
            sector_results = connection.execute(query)
            
            sectors = []
            avg_returns = []
            stock_counts = []
            
            for row in sector_results:
                sectors.append(row[0])
                avg_returns.append(float(row[1]))
                stock_counts.append(int(row[2]))
                
            if sectors and avg_returns:
                st.subheader("Average Yearly Return by Sector")
                
                # Create and display the results table
                sector_performance_df = pd.DataFrame({
                    'Sector': sectors,
                    'Average Yearly Return (%)': avg_returns,
                    'Number of Stocks': stock_counts
                })
                
                # Format the percentage column
                sector_performance_df['Average Yearly Return (%)'] = sector_performance_df['Average Yearly Return (%)'].map('{:.2f}%'.format)
                sector_performance_df.index = sector_performance_df.index + 1
                
                # Display without the stock count column
                st.table(sector_performance_df[['Sector', 'Average Yearly Return (%)']])
                
                # Create visualization
                fig_sector, ax_sector = plt.subplots(figsize=(12, 6))
                colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in avg_returns]
                bars = ax_sector.bar(sectors, avg_returns, color=colors, width=0.6)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax_sector.text(
                        bar.get_x() + bar.get_width()/2., 
                        height,
                        f'{height:.1f}%',
                        ha='center', 
                        va='bottom' if height >= 0 else 'top',
                        fontsize=10
                    )

                ax_sector.set_title('Average Yearly Return by Sector', fontsize=14, pad=20)
                ax_sector.set_ylabel('Return (%)', fontsize=12)
                ax_sector.set_xticks(range(len(sectors)))
                ax_sector.set_xticklabels(sectors, rotation=45, ha='right', fontsize=10)
                ax_sector.tick_params(axis='y', labelsize=10)
                ax_sector.grid(axis='y', linestyle=':', alpha=0.7)
                
                # Remove spines
                for spine in ['top', 'right']:
                    ax_sector.spines[spine].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig_sector)
            else:
                st.warning("No sector data available for plotting")
                
    except Exception as e:
        st.error(f"Error calculating sector performance: {e}")

def task4_stock_correlation(engine, table_names):
    """Display stock price correlation analysis using SQL queries"""
    st.subheader("Stock Price Correlation Analysis")
    
    if not table_names:
        st.error("No tables found in the database.")
        return
    
    # All stocks correlation
    try:
        with engine.connect() as connection:
            # Create temporary table for all prices (with IF NOT EXISTS)
            connection.execute(text("""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_all_prices (
                    date DATE,
                    symbol VARCHAR(50),
                    close DECIMAL(20, 4),
                    PRIMARY KEY (date, symbol)
                )
            """))
            
            # Clear existing data
            connection.execute(text("TRUNCATE TABLE temp_all_prices"))
            
            # Insert data from all tables
            for table in table_names:
                try:
                    connection.execute(
                        text(f"INSERT INTO temp_all_prices (date, symbol, close) SELECT date, symbol, close FROM `{table}`"),
                    )
                except Exception as e:
                    st.warning(f"Error processing table {table}: {e}")
                    continue
            
            # Get the pivot table of closing prices
            query = text("SELECT date, symbol, close FROM temp_all_prices ORDER BY date")
            price_df = pd.read_sql(query, connection)
            
            if not price_df.empty:
                price_df['date'] = pd.to_datetime(price_df['date']).dt.date
                price_pivot = price_df.pivot(index='date', columns='symbol', values='close')
                correlation_matrix = price_pivot.corr(method='pearson')
                
                st.subheader("Stock Price Correlation Heatmap (All Stocks)")
                fig_corr, ax_corr = plt.subplots(figsize=(30, 30))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                plt.title('Stock Price Correlation Heatmap (All Stocks)')
                plt.tight_layout()
                st.pyplot(fig_corr)
            else:
                st.warning("Insufficient price data for correlation analysis (All Stocks).")
                
    except Exception as e:
        st.error(f"Error calculating all stocks correlation: {e}")
    
    # Sector-wise correlation
    st.subheader("Stock Price Correlation Heatmap (Sector-wise)")
    
    try:
        with engine.connect() as connection:
            # Get all unique sectors
            sectors = set()
            for table in table_names:
                query = text(f"SELECT sector FROM `{table}` LIMIT 1")
                result = connection.execute(query)
                row = result.fetchone()
                if row and row[0]:
                    sectors.add(row[0])
            
            if sectors:
                sector_tabs = st.tabs(list(sectors))
                
                for tab_index, sector in enumerate(sectors):
                    with sector_tabs[tab_index]:
                        try:
                            # Create temporary table for sector prices (with IF NOT EXISTS)
                            connection.execute(text("""
                                CREATE TEMPORARY TABLE IF NOT EXISTS temp_sector_prices (
                                    date DATE,
                                    symbol VARCHAR(50),
                                    close DECIMAL(20, 4),
                                    PRIMARY KEY (date, symbol)
                                )
                            """))
                            
                            # Clear existing data
                            connection.execute(text("TRUNCATE TABLE temp_sector_prices"))
                            
                            # Get all symbols in this sector
                            symbols_in_sector = []
                            for table in table_names:
                                query = text(f"""
                                    SELECT symbol FROM `{table}` 
                                    WHERE sector = :sector 
                                    LIMIT 1
                                """)
                                result = connection.execute(query, {'sector': sector})
                                row = result.fetchone()
                                if row:
                                    symbols_in_sector.append(row[0])
                            
                            if len(symbols_in_sector) > 1:
                                # Insert data for stocks in this sector
                                for symbol in symbols_in_sector:
                                    try:
                                        connection.execute(
                                            text(f"INSERT INTO temp_sector_prices (date, symbol, close) SELECT date, symbol, close FROM `{symbol}`"),
                                        )
                                    except Exception as e:
                                        st.warning(f"Error processing stock {symbol}: {e}")
                                        continue
                                
                                # Get the pivot table
                                query = text("SELECT date, symbol, close FROM temp_sector_prices ORDER BY date")
                                sector_price_df = pd.read_sql(query, connection)
                                
                                if not sector_price_df.empty:
                                    sector_price_df['date'] = pd.to_datetime(sector_price_df['date']).dt.date
                                    sector_price_pivot = sector_price_df.pivot(index='date', columns='symbol', values='close')
                                    sector_correlation_matrix = sector_price_pivot.corr(method='pearson')
                                    
                                    st.subheader(f"Correlation Heatmap for {sector} Sector")
                                    fig_sector_corr, ax_sector_corr = plt.subplots(figsize=(15, 15))
                                    sns.heatmap(sector_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_sector_corr)
                                    plt.title(f'Stock Price Correlation Heatmap for {sector} Sector')
                                    plt.tight_layout()
                                    st.pyplot(fig_sector_corr)
                                else:
                                    st.info(f"No price data available for {sector} sector.")
                            else:
                                st.info(f"Insufficient stock data (less than 2 stocks) to calculate correlation for the {sector} sector.")
                                
                        except Exception as e:
                            st.error(f"Error processing {sector} sector: {e}")
                            continue
            else:
                st.info("No sector information available for sector-wise correlation analysis.")
                
    except Exception as e:
        st.error(f"Error calculating sector-wise correlation: {e}")
    
    # Between sectors correlation (using average prices)
    st.subheader("Correlation Heatmap (Between Sectors - Average Price)")
    
    try:
        with engine.connect() as connection:
            # Get all unique sectors
            sectors = set()
            for table in table_names:
                query = text(f"SELECT sector FROM `{table}` LIMIT 1")
                result = connection.execute(query)
                row = result.fetchone()
                if row and row[0]:
                    sectors.add(row[0])
            
            if sectors:
                # Create temporary table for sector average prices
                connection.execute(text("""
                    CREATE TEMPORARY TABLE IF NOT EXISTS temp_sector_avg_prices (
                        date DATE,
                        sector VARCHAR(100),
                        avg_price DECIMAL(20, 4),
                        PRIMARY KEY (date, sector)
                    )
                """))
                
                # Clear existing data
                connection.execute(text("TRUNCATE TABLE temp_sector_avg_prices"))
                
                # Calculate and insert average prices for each sector
                for sector in sectors:
                    # Get all symbols in this sector
                    symbols_in_sector = []
                    for table in table_names:
                        query = text(f"""
                            SELECT symbol FROM `{table}` 
                            WHERE sector = :sector 
                            LIMIT 1
                        """)
                        result = connection.execute(query, {'sector': sector})
                        row = result.fetchone()
                        if row:
                            symbols_in_sector.append(row[0])
                    
                    if symbols_in_sector:
                        # Create dynamic parts for UNION ALL query
                        union_parts = []
                        params = {}
                        for i, symbol in enumerate(symbols_in_sector):
                            union_parts.append(f"SELECT date, :sector_{i} as sector, close FROM `{symbol}`")
                            params[f'sector_{i}'] = sector
                        
                        union_query = " UNION ALL ".join(union_parts)
                        
                        query = text(f"""
                            INSERT INTO temp_sector_avg_prices (date, sector, avg_price)
                            SELECT date, sector, AVG(close) as avg_price
                            FROM (
                                {union_query}
                            ) AS combined
                            GROUP BY date, sector
                        """)
                        connection.execute(query, params)
                
                # Get the pivot table of sector average prices
                query = text("SELECT date, sector, avg_price FROM temp_sector_avg_prices ORDER BY date")
                sector_avg_df = pd.read_sql(query, connection)
                
                if not sector_avg_df.empty:
                    sector_avg_df['date'] = pd.to_datetime(sector_avg_df['date']).dt.date
                    sector_avg_pivot = sector_avg_df.pivot(index='date', columns='sector', values='avg_price')
                    sector_correlation_matrix = sector_avg_pivot.corr(method='pearson')
                    
                    fig_sector_corr_overall, ax_sector_corr_overall = plt.subplots(figsize=(15, 15))
                    sns.heatmap(sector_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_sector_corr_overall)
                    plt.title('Correlation Heatmap (Between Sectors - Average Price)')
                    plt.tight_layout()
                    st.pyplot(fig_sector_corr_overall)
                else:
                    st.info("Insufficient sector data or overlapping dates to calculate correlation between sectors.")
            else:
                st.info("No sector information available for sector correlation analysis.")
                
    except Exception as e:
        st.error(f"Error calculating between-sector correlation: {e}")

def task5_monthly_performance(engine, table_names):
    """Display monthly performance analysis"""
    st.subheader("Monthly Performance Analysis")
    
    monthly_returns_data = {}
    
    for table in table_names:
        try:
            with engine.connect() as connection:
                # Query to get date and close price for each stock
                query = text(f"""
                    SELECT date, close 
                    FROM `{table}` 
                    ORDER BY date
                """)
                df = pd.read_sql(query, connection)
                
                if not df.empty:
                    symbol = table  # Assuming table name is the stock symbol
                    df['date'] = pd.to_datetime(df['date'])
                    df['year_month'] = df['date'].dt.to_period('M')
                    
                    # Get last close price for each month
                    monthly_prices = df.groupby('year_month')['close'].last()
                    
                    # Calculate monthly returns
                    monthly_returns = monthly_prices.pct_change().dropna()
                    monthly_returns_data[symbol] = monthly_returns
                    
        except Exception as e:
            st.warning(f"Error processing table {table}: {e}")
            continue

    if monthly_returns_data:
        all_months = sorted(list(set(month for returns in monthly_returns_data.values() for month in returns.index)))

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Monthly Performance", "Top Performers Analysis"])

        with tab1:
            st.subheader("Monthly Performance Breakdown")
            for month in all_months:
                month_data = []
                for symbol, returns in monthly_returns_data.items():
                    if month in returns:
                        month_data.append({'symbol': symbol, 'return': returns[month]})

                if month_data:
                    top_n = sorted(month_data, key=lambda x: x['return'], reverse=True)[:5]
                    bottom_n = sorted(month_data, key=lambda x: x['return'])[:5]

                    st.subheader(f"**{month.strftime('%Y-%m')}**")

                    # Table for Top Gainers
                    st.write("**Top 5 Gainers:**")
                    top_gainers_df = pd.DataFrame(top_n)
                    top_gainers_df['return'] = top_gainers_df['return'].map('{:.2%}'.format)
                    top_gainers_df.index = range(1, len(top_gainers_df) + 1)
                    st.table(top_gainers_df)

                    # Table for Top Losers
                    st.write("**Top 5 Losers:**")
                    top_losers_df = pd.DataFrame(bottom_n)
                    top_losers_df['return'] = top_losers_df['return'].map('{:.2%}'.format)
                    top_losers_df.index = range(1, len(top_losers_df) + 1)
                    st.table(top_losers_df)

                    # Visualization
                    fig_month, (ax_top, ax_bottom) = plt.subplots(1, 2, figsize=(12, 4))
                    fig_month.suptitle(f"Performance for {month.strftime('%Y-%m')}", y=1.05)

                    # Top gainers plot
                    top_symbols = [x['symbol'] for x in top_n]
                    top_returns = [x['return'] * 100 for x in top_n]  # Convert to percentage
                    ax_top.bar(top_symbols, top_returns, color='green')
                    ax_top.set_title('Top 5 Gainers')
                    ax_top.set_ylabel('Return (%)')
                    ax_top.tick_params(axis='x', rotation=45, labelsize=8)

                    # Bottom losers plot
                    bottom_symbols = [x['symbol'] for x in bottom_n]
                    bottom_returns = [x['return'] * 100 for x in bottom_n]  # Convert to percentage
                    ax_bottom.bar(bottom_symbols, bottom_returns, color='red')
                    ax_bottom.set_title('Top 5 Losers')
                    ax_bottom.set_ylabel('Return (%)')
                    ax_bottom.tick_params(axis='x', rotation=45, labelsize=8)

                    plt.tight_layout()
                    st.pyplot(fig_month)
                    st.markdown("---") # Separator between months

        with tab2:
            st.subheader("Top Performers Analysis")

            # Aggregate which stocks appear most in top/bottom performers
            top_appearances = {}
            bottom_appearances = {}

            for month in all_months:
                month_data = []
                for symbol, returns in monthly_returns_data.items():
                    if month in returns:
                        month_data.append({'symbol': symbol, 'return': returns[month]})

                if month_data:
                    top_n = sorted(month_data, key=lambda x: x['return'], reverse=True)[:5]
                    bottom_n = sorted(month_data, key=lambda x: x['return'])[:5]

                    for item in top_n:
                        top_appearances[item['symbol']] = top_appearances.get(item['symbol'], 0) + 1

                    for item in bottom_n:
                        bottom_appearances[item['symbol']] = bottom_appearances.get(item['symbol'], 0) + 1

            # Show most frequent top and bottom performers in tables and charts
            if top_appearances and bottom_appearances:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Most Frequent Top Performers**")
                    top_frequent = sorted(top_appearances.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_freq_df = pd.DataFrame(top_frequent, columns=['Symbol', 'Number of Top 5 Appearances'])
                    top_freq_df.index = range(1, len(top_freq_df) + 1)
                    st.table(top_freq_df)

                    fig_top_freq = plt.figure()
                    plt.bar([x[0] for x in top_frequent], [x[1] for x in top_frequent], color='green')
                    plt.title('Stocks with Most Top 5 Appearances')
                    plt.ylabel('Number of Months in Top 5')
                    plt.xticks(rotation=45, ha='right', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_top_freq)

                with col2:
                    st.write("**Most Frequent Bottom Performers**")
                    bottom_frequent = sorted(bottom_appearances.items(), key=lambda x: x[1], reverse=True)[:10]
                    bottom_freq_df = pd.DataFrame(bottom_frequent, columns=['Symbol', 'Number of Bottom 5 Appearances'])
                    bottom_freq_df.index = range(1, len(bottom_freq_df) + 1)
                    st.table(bottom_freq_df)

                    fig_bottom_freq = plt.figure()
                    plt.bar([x[0] for x in bottom_frequent], [x[1] for x in bottom_frequent], color='red')
                    plt.title('Stocks with Most Bottom 5 Appearances')
                    plt.ylabel('Number of Months in Bottom 5')
                    plt.xticks(rotation=45, ha='right', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_bottom_freq)
            else:
                st.warning("Could not calculate frequent performers.")
    else:
        st.warning("Could not calculate monthly returns.")

# --- Main Application ---
def main():
    st.title("Data-Driven Stock Analysis")
    st.subheader("Explore Stock Market Insights")
    
    # Initialize database connection
    engine = create_engine(DATABASE_URL)
    
    # Get table names once at startup
    table_names = get_table_names()
    
    if not table_names:
        st.error("Failed to connect to database or no tables found.")
        return
    
    # Initialize session state for button click
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    
    # Connect button - now actually controls visibility
    if st.button("Load Stock Data"):
        st.session_state.show_analysis = True
        st.success("Data loaded successfully!")
    
    # Only show analysis if button was clicked
    if st.session_state.show_analysis:
        # Show market summary by default
        task0_market_summary(engine)
        
        # Create a dropdown for other analyses
        st.markdown("---")
        st.subheader("Additional Analyses")
        
        analysis_options = {
            "Market Summary": task0_market_summary,
            "Volatility Analysis": task1_volatility_analysis,
            "Cumulative Returns": task2_cumulative_returns,
            "Sector Performance": task3_sector_performance,
            "Stock Correlation": task4_stock_correlation,
            "Monthly Performance": task5_monthly_performance
        }
        
        selected_analysis = st.selectbox(
            "Select an analysis to view:",
            list(analysis_options.keys()),
            index=0
        )
        
        if selected_analysis != "Market Summary":
            analysis_function = analysis_options[selected_analysis]
            if selected_analysis == "Stock Correlation":
                analysis_function(engine, table_names)
            if selected_analysis == "Monthly Performance":
                analysis_function(engine, table_names)
            else:
                analysis_function(engine)
    else:
        st.info("Click the button above to load the analysis.")

    st.sidebar.info("This application performs various stock market analyses using data from your MySQL database.")

if __name__ == "__main__":
    main()