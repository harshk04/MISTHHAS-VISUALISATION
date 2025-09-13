import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Any
import configparser
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
from collections import Counter
import warnings
from decimal import Decimal
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from dotenv import load_dotenv
import plotly.express as px
import streamlit as st
import streamlit as st
import configparser
import os


# Load environment variables from .env
load_dotenv()
load_dotenv(dotenv_path=".env")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NL2SQLConfig:
    """Configuration management for NL2SQL system"""
    def __init__(self, config_file: str = 'config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        """Load configuration from secrets.toml"""
        try:
            self.config.read(self.config_file)
        except Exception:
            pass

        # Database configuration from secrets.toml
        self.db_connection_string = st.secrets["DB_CONNECTION_STRING"]

        # OpenAI API configuration from secrets.toml
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.model_name = st.secrets["OPENAI_MODEL_NAME"]
        print(f"Using OpenAI model: {self.model_name}")

class DatabaseManager:
    """Database connection and query execution management"""
    def __init__(self, config: NL2SQLConfig):
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection using connection string"""
        try:
            self.connection = psycopg2.connect(self.config.db_connection_string)
            logger.info("Successfully connected to PostgreSQL cloud database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL query safely with parameterized inputs using cursor approach"""
        if not self.connection:
            return [], "No database connection established"

        try:
            cur = self.connection.cursor()
            
            # Execute the query
            cur.execute(query, params)
            
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                # For SELECT queries, fetch all results
                rows = cur.fetchall()
                
                # Get column names
                colnames = [desc[0] for desc in cur.description]
                
                # Convert to list of dictionaries
                results = [dict(zip(colnames, row)) for row in rows]
                
                cur.close()
                return results, None
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE)
                self.connection.commit()
                cur.close()
                return [], None
                
        except Exception as e:
            self.connection.rollback()
            error_msg = f"Query execution error: {str(e)}"
            logger.error(error_msg)
            if 'cur' in locals():
                cur.close()
            return [], error_msg

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class SmartGraphGenerator:
    """Enhanced smart graph generation with improved logic for visualization strategy"""
    
    def __init__(self, output_dir: str = "graphs"):
        self.output_dir = output_dir
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _is_decimal_column(self, series: pd.Series) -> bool:
        """Check if a series contains Decimal objects from PostgreSQL"""
        try:
            sample = series.dropna().head(5)
            if len(sample) == 0:
                return False
            # Check if any non-null values are Decimal objects
            for value in sample:
                if isinstance(value, Decimal):
                    return True
            return False
        except:
            return False

    def _convert_decimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Decimal columns to float for proper numeric analysis"""
        df_converted = df.copy()
        for col in df.columns:
            if self._is_decimal_column(df[col]):
                logger.info(f"Converting Decimal column '{col}' to float")
                try:
                    df_converted[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
                    logger.info(f"Successfully converted '{col}' - new dtype: {df_converted[col].dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}': {e}")
        return df_converted

    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if a series contains actual date strings (more restrictive)"""
        try:
            sample = series.dropna().head(8)
            if len(sample) == 0:
                return False
            
            # Only check string-like values
            if not (series.dtype == 'object' or pd.api.types.is_string_dtype(series)):
                return False
            
            # Check if values look like date strings (contain date patterns)
            for value in sample:
                value_str = str(value).strip()
                # Skip if it's purely numeric
                try:
                    float(value_str)
                    return False # If it can be converted to float, it's not a date
                except:
                    pass
                
                # Must contain date separators
                if not any(sep in value_str for sep in ['-', '/', ' ', ':']):
                    return False
            
            # Try to convert to datetime
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False

    def analyze_dataframe_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame structure for smart visualization with improved classification"""
        logger.info(f"\n=== ANALYZING DATAFRAME ===")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Convert Decimal columns to float BEFORE analysis
        df_converted = self._convert_decimal_columns(df)
        
        analysis = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'id_columns': [],
            'column_details': {}
        }
        
        for col in df_converted.columns:
            logger.info(f"\nAnalyzing column: {col}")
            logger.info(f" Original dtype: {df[col].dtype}")
            logger.info(f" Converted dtype: {df_converted[col].dtype}")
            logger.info(f" Unique values: {df_converted[col].nunique()}")
            logger.info(f" Sample values: {df_converted[col].dropna().head(8).tolist()}")
            
            col_lower = str(col).lower()
            series = df_converted[col]
            
            # Column details
            analysis['column_details'][col] = {
                'dtype': str(series.dtype),
                'unique_count': series.nunique(),
                'sample_values': series.dropna().head(8).tolist(),
                'is_numeric': pd.api.types.is_numeric_dtype(series),
                'is_string': pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(series)
            }
            
            # IMPROVED CLASSIFICATION LOGIC
            if pd.api.types.is_numeric_dtype(series):
                # More specific ID column detection
                is_likely_id = self._is_likely_id_column(col, series, df_converted)
                if is_likely_id:
                    analysis['id_columns'].append(col)
                    logger.info(f" → Classified as: ID/IDENTIFIER")
                else:
                    analysis['numeric_columns'].append(col)
                    logger.info(f" → Classified as: NUMERIC")
            
            elif pd.api.types.is_datetime64_any_dtype(series) or self._looks_like_date(series):
                analysis['datetime_columns'].append(col)
                logger.info(f" → Classified as: DATETIME")
            
            else:
                # String/categorical column
                analysis['categorical_columns'].append(col)
                logger.info(f" → Classified as: CATEGORICAL")
        
        logger.info(f"\n=== CLASSIFICATION RESULTS ===")
        logger.info(f"Numeric columns: {analysis['numeric_columns']}")
        logger.info(f"Categorical columns: {analysis['categorical_columns']}")
        logger.info(f"DateTime columns: {analysis['datetime_columns']}")
        logger.info(f"ID columns: {analysis['id_columns']}")
        
        return analysis

    def _is_likely_id_column(self, col_name: str, series: pd.Series, df: pd.DataFrame) -> bool:
        """Improved logic to determine if a column is likely an ID/identifier"""
        col_lower = col_name.lower()
        
        # Strong indicators for ID columns
        id_keywords = ['id', 'key', 'pk', 'code', 'number', 'no']
        
        # Check if column name explicitly suggests it's an ID
        name_suggests_id = any(keyword in col_lower for keyword in id_keywords)
        
        # Check if it's likely a sequential ID (integers starting from 1 or similar)
        is_sequential_like = False
        if pd.api.types.is_integer_dtype(series):
            min_val = series.min()
            max_val = series.max()
            unique_count = series.nunique()
            total_count = len(df)
            
            # Sequential-like: starts low, mostly unique, reasonable range
            is_sequential_like = (
                min_val >= 1 and
                max_val <= total_count * 2 and  # Not too spread out
                unique_count / total_count > 0.95  # Very high uniqueness (95%+)
            )
        
        # Business logic: Only classify as ID if name suggests it OR it's clearly sequential
        return name_suggests_id or is_sequential_like

    def determine_visualization_strategy(self, analysis: Dict[str, Any], df: pd.DataFrame, query: str = "") -> List[Dict[str, Any]]:
        """Enhanced visualization strategy with smarter logic and query context awareness"""
        strategies = []
        
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        datetime_cols = analysis['datetime_columns']
        
        logger.info(f"\n=== DETERMINING VISUALIZATION STRATEGY ===")
        logger.info(f"Query: {query}")
        logger.info(f"Data shape: {df.shape}")
        
        # Check if we have the right combination for multi-series time series
        has_datetime = len(datetime_cols) > 0
        has_categorical = len(categorical_cols) > 0
        has_numeric = len(numeric_cols) > 0
        
        if has_datetime:
            datetime_col = datetime_cols[0]
            unique_dates = df[datetime_col].nunique()
            logger.info(f"DateTime column '{datetime_col}' has {unique_dates} unique values")
            
            # Strategy 1A: Multi-series time series (datetime + categorical + numeric)
            # This is perfect for "monthly sales by category" type queries
            if has_categorical and has_numeric and unique_dates >= 2:
                strategies.append({
                    'type': 'multi_series_time_series',
                    'x_column': datetime_col,
                    'y_column': numeric_cols[0],
                    'category_column': categorical_cols[0],
                    'priority': 1
                })
                logger.info("Added multi-series time series strategy (priority 1)")
            
            # Strategy 1B: Simple time series (datetime + numeric) - only if enough dates
            elif has_numeric and unique_dates >= 3:
                strategies.append({
                    'type': 'time_series',
                    'x_column': datetime_col,
                    'y_columns': numeric_cols[:3],  # Limit to 3 lines max
                    'priority': 2
                })
                logger.info("Added simple time series strategy (priority 2)")
        
        # Strategy 2: Bar chart (categorical + numeric) - Higher priority when time series doesn't make sense
        if has_categorical and has_numeric:
            priority = 1 if not has_datetime or (has_datetime and df[datetime_cols[0]].nunique() <= 2) else 3
            strategies.append({
                'type': 'bar_chart',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'priority': priority
            })
            logger.info(f"Added bar chart strategy (priority {priority})")
        
        # Strategy 3: Grouped bar chart (datetime + categorical + numeric, few dates)
        if has_datetime and has_categorical and has_numeric:
            unique_dates = df[datetime_cols[0]].nunique()
            unique_categories = df[categorical_cols[0]].nunique()
            if 2 <= unique_dates <= 6 and unique_categories <= 8:
                strategies.append({
                    'type': 'grouped_bar_chart',
                    'x_column': categorical_cols[0],
                    'y_column': numeric_cols[0],
                    'group_column': datetime_cols[0],
                    'priority': 2
                })
                logger.info("Added grouped bar chart strategy (priority 2)")
        
        # Strategy 4: Pie chart (categorical + numeric, small categories)
        if has_categorical and has_numeric:
            cat_col = categorical_cols[0]
            if analysis['column_details'][cat_col]['unique_count'] <= 8:
                strategies.append({
                    'type': 'pie_chart',
                    'category_column': cat_col,
                    'value_column': numeric_cols[0],
                    'priority': 4
                })
                logger.info("Added pie chart strategy (priority 4)")
        
        # Strategy 5: Scatter plot (multiple numeric)
        if len(numeric_cols) >= 2:
            strategies.append({
                'type': 'scatter_plot',
                'x_column': numeric_cols[1],
                'y_column': numeric_cols[0],
                'priority': 5
            })
            logger.info("Added scatter plot strategy (priority 5)")
        
        # Strategy 6: Histogram (single numeric)
        if has_numeric:
            strategies.append({
                'type': 'histogram',
                'column': numeric_cols[0],
                'priority': 6
            })
            logger.info("Added histogram strategy (priority 6)")
        
        # Strategy 7: Categorical count (categorical only)
        if has_categorical:
            strategies.append({
                'type': 'categorical_count',
                'column': categorical_cols[0],
                'priority': 7
            })
            logger.info("Added categorical count strategy (priority 7)")
        
        # Sort by priority
        strategies.sort(key=lambda x: x['priority'])
        
        logger.info(f"\n=== FINAL VISUALIZATION STRATEGIES ===")
        for i, strategy in enumerate(strategies):
            logger.info(f"{i+1}. {strategy['type']} (priority {strategy['priority']}): {strategy}")
        
        return strategies

    def create_multi_series_time_series(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create multi-series time series plot (e.g., sales by category over time)"""
        try:
            logger.info("\n=== CREATING MULTI-SERIES TIME SERIES WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            x_col = strategy['x_column']  # datetime
            y_col = strategy['y_column']  # numeric value
            cat_col = strategy['category_column']  # category for different lines
            
            logger.info(f"X-axis (datetime): {x_col}")
            logger.info(f"Y-axis (numeric): {y_col}")
            logger.info(f"Category (lines): {cat_col}")
            
            # Prepare the data
            df_plot = df_converted[[x_col, y_col, cat_col]].copy()
            df_plot = df_plot.dropna()
            
            # Convert datetime
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
            df_plot = df_plot.sort_values([x_col, cat_col])
            
            logger.info(f"Data shape after cleaning: {df_plot.shape}")
            logger.info(f"Unique categories: {df_plot[cat_col].nunique()}")
            logger.info(f"Date range: {df_plot[x_col].min()} to {df_plot[x_col].max()}")
            
            # Group by datetime and category, sum values (in case of duplicates)
            grouped_data = df_plot.groupby([x_col, cat_col])[y_col].sum().reset_index()
            
            fig = go.Figure()
            
            # Create a line for each category
            categories = sorted(grouped_data[cat_col].unique())
            colors = px.colors.qualitative.Set3[:len(categories)]
            
            for i, category in enumerate(categories):
                cat_data = grouped_data[grouped_data[cat_col] == category]
                fig.add_trace(go.Scatter(
                    x=cat_data[x_col],
                    y=cat_data[y_col],
                    mode='lines+markers',
                    name=str(category),
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                height=600,
                font=dict(size=12),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/multi_series_time_series_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Multi-series time series saved successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating multi-series time series: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def create_grouped_bar_chart(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create grouped bar chart (e.g., sales by category grouped by month)"""
        try:
            logger.info("\n=== CREATING GROUPED BAR CHART WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            x_col = strategy['x_column']  # categorical (e.g., category)
            y_col = strategy['y_column']  # numeric value
            group_col = strategy['group_column']  # grouping (e.g., month)
            
            logger.info(f"X-axis (categorical): {x_col}")
            logger.info(f"Y-axis (numeric): {y_col}")
            logger.info(f"Grouping: {group_col}")
            
            # Prepare the data
            chart_data = df_converted[[x_col, y_col, group_col]].copy()
            chart_data = chart_data.dropna()
            
            # Format group column for better display
            if pd.api.types.is_datetime64_any_dtype(chart_data[group_col]):
                chart_data[group_col] = pd.to_datetime(chart_data[group_col]).dt.strftime('%Y-%m')
            
            # Group and sum
            grouped_data = chart_data.groupby([x_col, group_col])[y_col].sum().reset_index()
            
            # Create grouped bar chart
            fig = px.bar(
                grouped_data,
                x=x_col,
                y=y_col,
                color=group_col,
                title=title,
                barmode='group'
            )
            
            fig.update_layout(
                height=600,
                font=dict(size=12)
            )
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/grouped_bar_chart_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Grouped bar chart saved successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating grouped bar chart: {e}")
            return None

    def create_bar_chart_with_values(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create bar chart using ACTUAL VALUES with Plotly"""
        try:
            logger.info(f"\n=== CREATING BAR CHART WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            x_col = strategy['x_column']
            y_col = strategy['y_column']
            
            logger.info(f"X-axis (categorical): {x_col}")
            logger.info(f"Y-axis (numeric): {y_col}")
            
            # Prepare the data - CRITICAL: Use actual values from converted DataFrame
            chart_data = df_converted[[x_col, y_col]].copy()
            
            # Remove any null values
            chart_data = chart_data.dropna()
            
            logger.info(f"Data shape after cleaning: {chart_data.shape}")
            logger.info(f"Sample data:\n{chart_data.head()}")
            
            # Group by category and sum the values (in case of duplicates)
            if chart_data[x_col].duplicated().any():
                logger.info("Found duplicate categories, aggregating...")
                grouped_data = chart_data.groupby(x_col)[y_col].sum().reset_index()
            else:
                grouped_data = chart_data
            
            logger.info(f"Final data for plotting:\n{grouped_data}")
            
            # Sort by values for better visualization
            grouped_data = grouped_data.sort_values(y_col, ascending=False)
            
            # Limit to top 15 for readability
            if len(grouped_data) > 15:
                grouped_data = grouped_data.head(15)
                title = f"{title} (Top 15)"
            
            logger.info(f"Plotting values: {grouped_data[y_col].values}")
            logger.info(f"Categories: {grouped_data[x_col].values}")
            
            # Create Plotly bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=grouped_data[x_col],
                    y=grouped_data[y_col],
                    text=[f'{val:,.0f}' if val >= 1000 else f'{val:.1f}' for val in grouped_data[y_col]],
                    textposition='auto',
                    marker_color='steelblue'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                showlegend=False,
                height=600,
                font=dict(size=12)
            )
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/bar_chart_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Bar chart saved successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def create_time_series(self, df: pd.DataFrame, strategy: dict, title: str) -> Optional[str]:
        """Create time series plot with Plotly with distinct line colors"""
        try:
            logger.info("\n=== CREATING TIME SERIES WITH PLOTLY ===")
            
            df_converted = self._convert_decimal_columns(df)
            x_col = strategy['x_column']
            y_cols = strategy['y_columns']
            
            df_plot = df_converted.copy()
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
            df_plot = df_plot.sort_values(x_col)
            
            fig = go.Figure()
            
            # Use a color palette with enough distinct colors
            colors = px.colors.qualitative.D3  # 10 visually distinct colors
            
            for i, y_col in enumerate(y_cols):
                fig.add_trace(go.Scatter(
                    x=df_plot[x_col],
                    y=df_plot[y_col],
                    mode='lines+markers',
                    name=y_col.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title='Value',
                height=600,
                font=dict(size=12)
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/time_series_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Time series saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating time series: {e}")
            return None

    def create_pie_chart(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create pie chart with actual values using Plotly"""
        try:
            logger.info("\n=== CREATING PIE CHART WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            cat_col = strategy['category_column']
            val_col = strategy['value_column']
            
            # Aggregate data
            pie_data = df_converted.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
            
            # Limit to top 8 slices
            if len(pie_data) > 8:
                top_data = pie_data.head(7)
                other_sum = pie_data.iloc[7:].sum()
                pie_data = pd.concat([top_data, pd.Series({'Others': other_sum})])
            
            fig = go.Figure(data=[go.Pie(
                labels=pie_data.index,
                values=pie_data.values,
                textinfo='label+percent'
            )])
            
            fig.update_layout(
                title=title,
                height=600,
                font=dict(size=12)
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/pie_chart_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Pie chart saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return None

    def create_scatter_plot(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create scatter plot with Plotly (without trend line)"""
        try:
            logger.info("\n=== CREATING SCATTER PLOT WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            x_col = strategy['x_column']
            y_col = strategy['y_column']
            
            fig = go.Figure()
            
            # Scatter points only
            fig.add_trace(go.Scatter(
                x=df_converted[x_col],
                y=df_converted[y_col],
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.7),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                height=600,
                font=dict(size=12)
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/scatter_plot_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Scatter plot saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return None

    def create_histogram(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create histogram with Plotly (mean shown as vertical line; median shown as a value-only annotation)."""
        try:
            logger.info("\n=== CREATING HISTOGRAM WITH PLOTLY ===")
            
            # Convert Decimal columns first
            df_converted = self._convert_decimal_columns(df)
            
            col = strategy['column']
            data = df_converted[col].dropna()
            
            if data.empty:
                logger.warning(f"No data available for column: {col}")
                return None
            
            nbins = min(25, max(1, len(data.unique())))
            
            fig = go.Figure(data=[go.Histogram(
                x=data,
                nbinsx=nbins,
                marker_color='steelblue',
                opacity=0.7
            )])
            
            fig.add_annotation(
                y=1.02,  # place above the plotting area
                yref='paper',  # use paper coords so annotation is placed relative to the figure
                showarrow=False,
                font=dict(color='orange', size=12),
                align='center'
            )
            
            fig.update_layout(
                title=title or f'Distribution of {col.replace("_", " ").title()}',
                xaxis_title=col.replace('_', ' ').title(),
                yaxis_title='Frequency',
                height=600,
                font=dict(size=12),
                margin=dict(t=80)  # give space for the annotation on top
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/histogram_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Histogram saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return None

    def create_categorical_count(self, df: pd.DataFrame, strategy: Dict[str, Any], title: str) -> Optional[str]:
        """Create categorical count chart with Plotly"""
        try:
            logger.info("\n=== CREATING CATEGORICAL COUNT WITH PLOTLY ===")
            
            col = strategy['column']
            data = df[col].value_counts().head(15)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=data.index,
                    y=data.values,
                    text=data.values,
                    textposition='auto',
                    marker_color='steelblue'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title=col.replace('_', ' ').title(),
                yaxis_title='Count',
                height=600,
                font=dict(size=12),
                showlegend=False
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/categorical_count_{timestamp}.html"
            fig.write_html(filename)
            
            logger.info(f"Categorical count chart saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating categorical count chart: {e}")
            return None

    def generate_smart_visualization(self, data: List[Dict], query: str) -> Optional[str]:
        """Generate smart visualization with enhanced strategy logic"""
        if not data:
            logger.warning("No data provided for visualization")
            return None
        
        try:
            logger.info(f"\nSTARTING SMART VISUALIZATION WITH PLOTLY")
            logger.info(f"Query: {query}")
            logger.info(f"Data records: {len(data)}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Analyze structure (now with proper classification)
            analysis = self.analyze_dataframe_structure(df)
            
            # Determine strategies with enhanced logic
            strategies = self.determine_visualization_strategy(analysis, df, query)
            
            if not strategies:
                logger.warning("No visualization strategies found")
                return None
            
            title = query[:60] + "..." if len(query) > 60 else query
            
            # Try each strategy until one succeeds
            for i, strategy in enumerate(strategies):
                logger.info(f"\nTrying strategy {i+1}: {strategy['type']}")
                try:
                    result = None
                    if strategy['type'] == 'multi_series_time_series':
                        result = self.create_multi_series_time_series(df, strategy, title)
                    elif strategy['type'] == 'grouped_bar_chart':
                        result = self.create_grouped_bar_chart(df, strategy, title)
                    elif strategy['type'] == 'bar_chart':
                        result = self.create_bar_chart_with_values(df, strategy, title)
                    elif strategy['type'] == 'time_series':
                        result = self.create_time_series(df, strategy, title)
                    elif strategy['type'] == 'pie_chart':
                        result = self.create_pie_chart(df, strategy, title)
                    elif strategy['type'] == 'scatter_plot':
                        result = self.create_scatter_plot(df, strategy, title)
                    elif strategy['type'] == 'histogram':
                        result = self.create_histogram(df, strategy, title)
                    elif strategy['type'] == 'categorical_count':
                        result = self.create_categorical_count(df, strategy, title)
                    
                    if result:
                        logger.info(f"SUCCESS! Generated {strategy['type']}: {result}")
                        return result
                except Exception as e:
                    logger.error(f"Strategy {strategy['type']} failed: {e}")
                    continue
            
            logger.error("All visualization strategies failed")
            return None
            
        except Exception as e:
            logger.error(f"Critical error in smart visualization: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

class SchemaManager:
    """Manage database schema information"""
    
    def __init__(self):
        self.ddl_schema = """
-- Dimension Tables
CREATE TABLE dim_bank_account (
    bank_account_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100),
    account_no VARCHAR(30),
    branch VARCHAR(50)
);

CREATE TABLE dim_category (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50)
);

CREATE TABLE dim_outlet (
    outlet_id SERIAL PRIMARY KEY,
    outlet_name VARCHAR(100),
    location VARCHAR(100),
    region VARCHAR(50)
);

CREATE TABLE dim_item (
    item_id SERIAL PRIMARY KEY,
    item_name VARCHAR(100),
    category_id INTEGER REFERENCES dim_category(category_id),
    uom VARCHAR(20)
);

CREATE TABLE dim_vendor (
    vendor_id SERIAL PRIMARY KEY,
    vendor_name VARCHAR(100),
    contact_no VARCHAR(20),
    city VARCHAR(50)
);

CREATE TABLE dim_employee (
    employee_id SERIAL PRIMARY KEY,
    employee_name VARCHAR(100),
    designation VARCHAR(50),
    department VARCHAR(50),
    joining_date DATE
);

-- Fact Tables
CREATE TABLE fact_cheques (
    cheque_id SERIAL PRIMARY KEY,
    bank_account_id INTEGER REFERENCES dim_bank_account(bank_account_id),
    cheque_no VARCHAR(30),
    amount NUMERIC(12, 2),
    issue_date DATE,
    clearing_status VARCHAR(10), -- 'Pending', 'Cleared'
    clearing_date DATE
);

CREATE TABLE fact_payables (
    vendor_id INTEGER REFERENCES dim_vendor(vendor_id),
    invoice_id SERIAL PRIMARY KEY,
    invoice_date DATE,
    due_date DATE,
    amount_due NUMERIC(12, 2),
    status VARCHAR(10) -- 'Paid', 'Unpaid', 'Partial'
);

CREATE TABLE fact_sales (
    sale_id SERIAL PRIMARY KEY,
    date DATE,
    outlet_id INTEGER REFERENCES dim_outlet(outlet_id),
    item_id INTEGER REFERENCES dim_item(item_id),
    category_id INTEGER REFERENCES dim_category(category_id),
    quantity INTEGER,
    gross_amount NUMERIC(12, 2),
    discount_amount NUMERIC(12, 2),
    net_amount NUMERIC(12, 2),
    payment_method VARCHAR(20) -- 'Cash', 'Card', 'UPI', 'Wallet'
);

CREATE TABLE fact_purchases (
    purchase_id SERIAL PRIMARY KEY,
    po_id INTEGER,
    date DATE,
    item_id INTEGER REFERENCES dim_item(item_id),
    vendor_id INTEGER REFERENCES dim_vendor(vendor_id),
    ordered_qty INTEGER,
    received_qty INTEGER,
    unit_price NUMERIC(12, 2),
    total_amount NUMERIC(12, 2),
    status VARCHAR(20) -- 'Ordered', 'Partially Received', 'Completed'
);

CREATE TABLE fact_attendance (
    attendance_id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES dim_employee(employee_id),
    date DATE,
    in_time TIME,
    out_time TIME,
    present_flag CHAR(1), -- 'Y' or 'N'
    remarks VARCHAR(50)
);
"""

    def get_schema_context(self) -> str:
        """Get formatted schema context for LLM"""
        context = f"""
You are an expert Business Data Analyst that converts natural language questions into optimized SQL queries and determines the best type of visualization.

Database Schema Information:
{self.ddl_schema}

Important Guidelines:
1. Always understand the user's intent carefully before generating SQL.
2. Use proper table joins when querying multiple tables (respect primary/foreign key relationships).
3. All monetary amounts are stored as NUMERIC(12,2). Show values in standard (indian INR) currency format where needed.
4. Dates are stored as DATE type (YYYY-MM-DD format). Use proper date functions for filtering and grouping.
5. For attendance queries, calculate percentages using present_flag ('Y' for present, 'N' for absent).
6. Always use appropriate aggregations (SUM, AVG, COUNT, MIN, MAX) as per the question.
7. Ensure that column aliases are human-friendly (e.g., 'total_sales' instead of 'Sales').
8. If the query involves time-series data, ORDER BY date/month/year appropriately.
9. Return only valid SQL compatible with PostgreSQL.

Your output must strictly include:
- SQL Query
"""
        return context

class OpenAINL2SQL:
    """OpenAI API integration for Natural Language to SQL conversion with enhanced English context"""
    
    def __init__(self, config: NL2SQLConfig):
        self.config = config
        self.model_name = config.model_name
        self._initialize_openai()

    def _initialize_openai(self):
        """Initialize OpenAI API connection"""
        try:
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = OpenAI(api_key=self.config.openai_api_key)
            logger.info(f"OpenAI initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {e}")
            raise

    def check_query_relevance(self, natural_language_query: str) -> Tuple[bool, str]:
        """Check if the user query is relevant to business data analysis"""
        prompt = f"""
Classify if the user query is about business data or not.

RELEVANT = anything mentioning outlets, locations, products, sales, revenue, expenses, reports, analytics, customers, vendors, employees, or database information.
NOT_RELEVANT = greetings, jokes, casual talk, or non-business topics.

Query: "{natural_language_query}"

Reply with:
1. "RELEVANT" or "NOT_RELEVANT"
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a query relevance classifier for business data systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                n=1,
            )
            
            response_text = response.choices[0].message.content.strip()
            lines = response_text.split('\n')
            relevance_status = lines[0].strip()
            explanation = lines[1] if len(lines) > 1 else "No explanation provided"
            
            is_relevant = relevance_status == "RELEVANT"
            
            logger.info(f"Query relevance check: {relevance_status}")
            logger.info(f"Explanation: {explanation}")
            
            return is_relevant, explanation
            
        except Exception as e:
            logger.error(f"Error checking query relevance: {e}")
            # If there's an error with the API call, assume it's relevant to avoid blocking legitimate queries
            return True, "Could not determine relevance due to API error. Proceeding with query processing."

    def generate_sql(self, natural_language_query: str, schema_context: str, conversation_context: str = "") -> Tuple[str, Optional[str]]:
        """Generate SQL query from natural language using OpenAI chat completion with follow-up context"""
        
        context_prompt = ""
        if conversation_context:
            context_prompt = f"""
Previous Conversation Context:
{conversation_context}

This may be a follow-up question related to the previous conversation. Use this context to better understand the current query.
"""
        
        prompt = f"""
You are an expert SQL developer working with a PostgreSQL database for a business ERP system.
Your task is to convert natural language queries into accurate SQL statements.

{context_prompt}

{schema_context}

REQUIREMENTS:
1. Generate SELECT statements for data retrieval queries
2. Always use proper JOINs when accessing multiple tables
3. Use appropriate WHERE clauses for filtering
4. Format dates properly (YYYY-MM-DD format)
5. Use ILIKE for case-insensitive string matching
6. Return only the SQL query - no explanations or markdown formatting
7. For percentage calculations, calculate them properly using COUNT and conditional logic
8. For attendance queries, use fact_attendance table with present_flag column
9. Consider standard business context and common practices
10. For follow-up questions, build upon previous context appropriately
11. Ensure all responses are in proper English

Natural Language Query: {natural_language_query}

Generate only the SQL query:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates SQL queries for business analytics with contextual understanding and clear English communication."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.8,
                max_tokens=2048,
                n=1,
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            print("\n=== GENERATED SQL QUERY ===\n", sql_query)
            return sql_query, None
            
        except Exception as e:
            error_msg = f"Error generating SQL with OpenAI: {str(e)}"
            logger.error(error_msg)
            return "", error_msg

    # === NEW METHOD: REGENERATE SQL AFTER ERROR ===
    def regenerate_sql_after_error(self, natural_language_query: str, failed_sql: str, 
                                 error_message: str, schema_context: str, 
                                 conversation_context: str = "") -> Tuple[str, Optional[str]]:
        """Regenerate SQL query after database execution error"""
        print("HIIIIIIIII",conversation_context )
        context_prompt = ""
        if conversation_context:
            context_prompt = f"""
Previous Conversation Context:
{conversation_context}

This may be a follow-up question related to the previous conversation.
"""
        
        # Create example of correct query structure based on your request
        example_query = """
Example of a correct query for "Net sales vs wastage quantity per item per outlet":
SELECT 
    s.item_id, 
    s.outlet_id, 
    SUM(s.net_amount) AS total_sales_amount, 
    SUM(w.wastage_qty) AS total_wastage_qty
FROM fact_sales s
LEFT JOIN fact_wastage w 
    ON s.item_id = w.item_id AND s.date = w.date AND s.outlet_id = w.outlet_id
GROUP BY s.item_id, s.outlet_id
ORDER BY s.item_id, s.outlet_id;
"""
        
        prompt = f"""
You are an expert SQL developer fixing a failed database query. The previous SQL query failed with an error.

{context_prompt}

{schema_context}

FAILED QUERY:
{failed_sql}

ERROR MESSAGE:
{error_message}

{example_query}

REQUIREMENTS FOR FIXING:
1. Analyze the error message carefully and understand what went wrong
2. Check if the table names, column names, and relationships are correct
3. Ensure proper JOIN syntax and conditions
4. Use appropriate aggregation functions (SUM, COUNT, AVG) when needed
5. Add GROUP BY clauses when using aggregation
6. Check for missing tables in the schema and use available ones
7. Use LEFT JOIN instead of INNER JOIN when one table might not have matching records
8. Return only the corrected SQL query - no explanations or markdown formatting
9. Ensure the query is syntactically correct and uses existing tables/columns only

Original Natural Language Query: {natural_language_query}

Generate the corrected SQL query:
"""
        print("HIIIIIIIII",context_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert SQL debugging assistant that fixes database query errors by analyzing error messages and schema information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.8,
                max_tokens=2048,
                n=1,
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            print(f"\n=== REGENERATED SQL QUERY (ATTEMPT TO FIX) ===\n{sql_query}")
            return sql_query, None
            
        except Exception as e:
            error_msg = f"Error regenerating SQL with OpenAI: {str(e)}"
            logger.error(error_msg)
            return "", error_msg

    def _is_safe_query_for_execution(self, sql_query: str) -> Tuple[bool, str]:
        """Validate that the generated SQL query is safe for execution"""
        if not sql_query:
            return False, "Empty query"
        
        query_upper = sql_query.upper().strip()
        
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            return False, "Query must start with SELECT or WITH"
        
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Dangerous keyword detected: {keyword}"
        
        return True, "Query is safe for execution"

    def generate_description(self, query_result: List[Dict], original_query: str, conversation_context: str = "") -> str:
        """Generate detailed natural language description of query results in proper English"""
        if not query_result:
            return "No data found matching your query criteria. This could be due to specific filters applied or no records matching the given time period."
        
        result_summary = f"Found {len(query_result)} records."
        
        if len(query_result) <= 5:
            result_text = json.dumps(query_result, indent=2, default=str)
        else:
            result_text = f"All records:\n{json.dumps(query_result, indent=2, default=str)}"
        
        context_prompt = ""
        if conversation_context:
            context_prompt = f"""
Previous Conversation Context:
{conversation_context}

This response may be building upon the previous conversation.
"""
        
        prompt = f"""
Provide a detailed, informative description (around 10 lines) that includes key findings and insights from the data, business implications and trends, specific numbers and percentages where relevant, recommendations or observations for business decision-making, all written in clear, professional English suitable for business executives. The tone should be professional yet conversational, focusing on actionable insights that can drive business decisions, ensuring language is grammatically correct and business-friendly. The analysis should be presented in paragraph format within 120 words, without bullet points. The context is aligned with the Indian market, so monetary values must be presented in INR (₹) with proper formatting.

{context_prompt}

Original Query: {original_query}
Results Summary: {result_summary}
Data: {result_text}

You are a professional business data analyst. Based on the provided sales data, give a detailed description of the results. Only provide the analysis of the data, no other information."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a detailed business analyst who provides comprehensive insights from the given data in clear, professional English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600,
                n=1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Query executed successfully. {result_summary} The data shows various business metrics that can help in making informed decisions for your organization."

class SmartNL2SQLProcessor:
    """Smart processor with enhanced graph generation, query relevance checking, and follow-up context"""
    
    def __init__(self, config_file: str = 'config.ini'):
        self.config = NL2SQLConfig(config_file)
        self.db_manager = DatabaseManager(self.config)
        self.schema_manager = SchemaManager()
        self.openai_nl2sql = OpenAINL2SQL(self.config)
        self.graph_generator = SmartGraphGenerator()

    # === UPDATED METHOD: ADDED RETRY LOGIC ===
    def process_query_with_smart_visualization(self, natural_language_query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Process query with enhanced visualization logic, relevance checking, follow-up context, and retry mechanism"""
        start_time = datetime.now()
        
        result = {
            'timestamp': start_time.isoformat(),
            'original_query': natural_language_query,
            'sql_query': '',
            'execution_results': [],
            'description': '',
            'graph_file': None,
            'graph_type': None,
            'error': None,
            'warning': None,
            'processing_time_seconds': 0,
            'query_safe': True,
            'is_relevant': True,
            'retry_attempts': 0,
            'retry_details': []
        }
        
        try:
            # Step 1: Check query relevance FIRST
            logger.info(f"Checking relevance for query: {natural_language_query}")
            is_relevant, relevance_explanation = self.openai_nl2sql.check_query_relevance(natural_language_query)
            result['is_relevant'] = is_relevant
            
            if not is_relevant:
                result['error'] = f"This query appears to be casual conversation rather than a business data request. {relevance_explanation}"
                result['description'] = "I'm here to help with business data analysis and reporting. Please ask questions about sales, employees, inventory, financials, or other business data."
                result['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
                return result
            
            # Step 2: Connect to database
            if not self.db_manager.connect():
                result['error'] = "Failed to connect to database"
                return result
            
            # Step 3: Generate SQL with context
            logger.info(f"Processing relevant business query: {natural_language_query}")
            schema_context = self.schema_manager.get_schema_context()
            
            sql_query, generation_error = self.openai_nl2sql.generate_sql(
                natural_language_query, schema_context, conversation_context
            )
            
            if generation_error:
                result['error'] = generation_error
                return result
            
            result['sql_query'] = sql_query
            
            # Step 4: Safety check
            is_safe, safety_message = self.openai_nl2sql._is_safe_query_for_execution(sql_query)
            result['query_safe'] = is_safe
            
            if not is_safe:
                result['warning'] = f"Query generated but not executed due to safety concerns: {safety_message}"
                result['description'] = "SQL query was generated but blocked from execution for security reasons."
                return result
            
            # === STEP 5: EXECUTE QUERY WITH RETRY LOGIC (NEW IMPLEMENTATION) ===
            execution_results = None
            execution_error = None
            current_sql = sql_query
            max_retries = 10
            
            for attempt in range(max_retries + 1):  # 0, 1, 2 (total 3 attempts)
                print(f"\n{'='*60}")
                print(f"ATTEMPT {attempt + 1} OF {max_retries + 1}")
                print(f"{'='*60}")
                print(f"Executing SQL: {current_sql}")
                
                execution_results, execution_error = self.db_manager.execute_query(current_sql)
                
                if execution_error is None:
                    # Success!
                    print(f"✅ SUCCESS on attempt {attempt + 1}")
                    result['execution_results'] = execution_results
                    break
                else:
                    # Failed - log the error
                    print(f"❌ FAILED on attempt {attempt + 1}")
                    print(f"ERROR: {execution_error}")
                    
                    result['retry_attempts'] = attempt + 1
                    result['retry_details'].append({
                        'attempt': attempt + 1,
                        'sql_query': current_sql,
                        'error': execution_error
                    })
                    
                    # If this was our last attempt, give up
                    if attempt >= max_retries:
                        print(f"❌ GIVING UP after {max_retries + 1} attempts")
                        result['error'] = f"Query failed after {max_retries + 1} attempts. Final error: {execution_error}"
                        return result
                    
                    # Try to regenerate the SQL
                    print(f"🔄 RETRYING - Regenerating SQL for attempt {attempt + 2}...")
                    print("HWIDBHB", natural_language_query)
                    print("HWIDBHB", current_sql)
                    print("HWIDBHB", execution_error)
                    print("HWIDBHB", schema_context)
                    print("HWIDBHB", conversation_context)
                    regenerated_sql, regeneration_error = self.openai_nl2sql.regenerate_sql_after_error(
                        natural_language_query, current_sql, execution_error, schema_context, conversation_context
                    )
                    
                    if regeneration_error:
                        print(f"❌ Failed to regenerate SQL: {regeneration_error}")
                        result['error'] = f"Failed to regenerate SQL after attempt {attempt + 1}: {regeneration_error}"
                        return result
                    
                    current_sql = regenerated_sql
                    result['sql_query'] = current_sql  # Update with the latest SQL
                    
                    # Safety check for regenerated query
                    is_safe, safety_message = self.openai_nl2sql._is_safe_query_for_execution(current_sql)
                    if not is_safe:
                        print(f"❌ Regenerated query failed safety check: {safety_message}")
                        result['error'] = f"Regenerated query failed safety check: {safety_message}"
                        return result
            
            # If we get here, the query executed successfully
            print(f"\n🎉 FINAL SUCCESS! Query executed successfully with {len(execution_results)} results")
            
            # Step 6: Enhanced smart visualization with improved logic
            if execution_results:
                logger.info(f"Generating smart visualization for {len(execution_results)} records")
                graph_file = self.graph_generator.generate_smart_visualization(
                    execution_results, natural_language_query
                )
                
                if graph_file:
                    result['graph_file'] = graph_file
                    result['graph_type'] = self._determine_graph_type_from_filename(graph_file)
                    logger.info(f"VISUALIZATION SUCCESS: {graph_file}")
                else:
                    logger.warning("No suitable visualization could be generated for this data")
            
            # Step 7: Generate detailed description with context
            description = self.openai_nl2sql.generate_description(
                execution_results, natural_language_query, conversation_context
            )
            result['description'] = description
            
            # Calculate processing time
            end_time = datetime.now()
            result['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"Query processed successfully in {result['processing_time_seconds']:.2f} seconds")
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error: {traceback.format_exc()}")
        finally:
            self.db_manager.close()
        
        return result

    def _determine_graph_type_from_filename(self, filename: str) -> str:
        """Extract graph type from filename"""
        if 'multi_series_time_series' in filename:
            return 'Multi-Series Time Series'
        elif 'grouped_bar_chart' in filename:
            return 'Grouped Bar Chart'
        elif 'time_series' in filename:
            return 'Time Series Plot'
        elif 'bar_chart' in filename:
            return 'Bar Chart'
        elif 'pie_chart' in filename:
            return 'Pie Chart'
        elif 'scatter_plot' in filename:
            return 'Scatter Plot'
        elif 'histogram' in filename:
            return 'Histogram'
        elif 'categorical_count' in filename:
            return 'Categorical Count Chart'
        else:
            return 'Data Visualization'



def main():
    """Main function with comprehensive test queries for different visualization types"""
    processor = SmartNL2SQLProcessor()
    
    # Enhanced diverse test queries designed to test improved visualization logic
    comprehensive_test_queries = [
        "Hello there",
        "How are you doing today?",
        "Show me the monthly net sales trend for each category in 2024",
        "Display total sales amount by category",
        "List employee count by department",
        "Compare sales performance across different regions",
        "What are the top performing products this year?",
        "Net sales vs wastage quantity per item per outlet"  # Added your example query
    ]
    
    print("\n" + "="*80)
    print("SMART NL2SQL PROCESSOR WITH RETRY MECHANISM AND ENGLISH LANGUAGE SUPPORT")
    print("="*80)
    
    all_results = []
    conversation_history = ""
    
    for i, query in enumerate(comprehensive_test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        result = processor.process_query_with_smart_visualization(query, conversation_history)
        all_results.append(result)
        
        # Build conversation context for follow-ups
        if result.get('is_relevant', True) and not result.get('error'):
            conversation_history += f"Query: {query}\nResult: {result.get('description', '')[:200]}...\n"
        
        # Handle different result types
        if not result.get('is_relevant', True):
            print(f"❌ IRRELEVANT QUERY")
            print(f"Response: {result['description']}")
        elif result['error']:
            print(f"❌ ERROR: {result['error']}")
            if result.get('retry_attempts', 0) > 0:
                print(f"🔄 RETRY ATTEMPTS: {result['retry_attempts']}")
                for retry_detail in result.get('retry_details', []):
                    print(f"   Attempt {retry_detail['attempt']}: {retry_detail['error']}")
        elif result['warning']:
            print(f"⚠️ WARNING: {result['warning']}")
            print(f" Generated SQL: {result['sql_query']}")
        else:
            print(f"✅ SUCCESS")
            print(f"Generated SQL: {result['sql_query']}")
            print(f"Results: {len(result['execution_results'])} records found")
            print(f"Description: {result['description']}")
            if result.get('retry_attempts', 0) > 0:
                print(f"🔄 Required {result['retry_attempts']} retries before success")
            
            if result.get('graph_file'):
                print(f"📊 GRAPH GENERATED: {result['graph_file']}")
                print(f"Graph Type: {result['graph_type']}")
            else:
                print(f"📊 NO GRAPH GENERATED")
        
        print("─" * 80)
    
    # Enhanced comprehensive summary
    successful_queries = [r for r in all_results if not r.get('error') and r.get('is_relevant', True)]
    irrelevant_queries = [r for r in all_results if not r.get('is_relevant', True)]
    blocked_queries = [r for r in all_results if r.get('warning')]
    failed_queries = [r for r in all_results if r.get('error') and r.get('is_relevant', True)]
    retry_queries = [r for r in all_results if r.get('retry_attempts', 0) > 0]
    graphs_generated = [r for r in all_results if r.get('graph_file')]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY WITH RETRY STATISTICS")
    print("="*80)
    print(f"Total queries processed: {len(all_results)}")
    print(f"Irrelevant queries detected: {len(irrelevant_queries)}")
    print(f"Successfully executed: {len(successful_queries)}")
    print(f"Failed after retries: {len(failed_queries)}")
    print(f"Required retries: {len(retry_queries)}")
    print(f"Blocked for security: {len(blocked_queries)}")
    print(f"Visualizations generated: {len(graphs_generated)}")
    
    if irrelevant_queries:
        print(f"\n🚫 IRRELEVANT QUERIES:")
        for r in irrelevant_queries:
            print(f" • '{r['original_query']}'")
    
    if retry_queries:
        print(f"\n🔄 QUERIES THAT REQUIRED RETRIES:")
        for r in retry_queries:
            print(f" • '{r['original_query']}' - {r['retry_attempts']} attempts")
    
    if successful_queries:
        avg_time = sum(r.get('processing_time_seconds', 0) for r in successful_queries) / len(successful_queries)
        print(f"\n⏱️ Average processing time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main()
