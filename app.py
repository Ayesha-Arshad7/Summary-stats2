import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from io import BytesIO
from fpdf import FPDF
import base64
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced EDA App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Advanced Exploratory Data Analysis App")

# Sidebar for analysis selection
st.sidebar.header("Analysis Options")
analysis_options = {
    "data_cleaning": st.sidebar.checkbox("Data Cleaning", value=True),
    "missing_values": st.sidebar.checkbox("Missing Values Analysis", value=True),
    "outlier_detection": st.sidebar.checkbox("Outlier Detection", value=True),
    "feature_engineering": st.sidebar.checkbox("Feature Engineering Suggestions", value=True),
    "correlation_analysis": st.sidebar.checkbox("Correlation Analysis", value=True),
    "distribution_analysis": st.sidebar.checkbox("Distribution Analysis", value=True),
    "time_series_analysis": st.sidebar.checkbox("Time Series Analysis", value=False),
    "group_comparisons": st.sidebar.checkbox("Group Comparisons", value=True)
}

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'summaries' not in st.session_state:
    st.session_state.summaries = []

# Function to add summary
def add_summary(category, description, findings, recommendation):
    st.session_state.summaries.append({
        "category": category,
        "description": description,
        "findings": findings,
        "recommendation": recommendation
    })

# Function to create downloadable PDF
def create_pdf(summaries):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EDA Analysis Summary", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    for i, summary in enumerate(summaries, 1):
        pdf.set_font("", "B", 14)
        pdf.cell(0, 10, f"{i}. {summary['category']}: {summary['description']}", 0, 1)
        pdf.set_font("", "", 12)
        pdf.multi_cell(0, 8, f"Findings: {summary['findings']}")
        pdf.multi_cell(0, 8, f"Recommendation: {summary['recommendation']}")
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')

# Function to analyze data types and basic info
def analyze_data_types(df):
    st.subheader("Data Types and Basic Information")
    
    # Create a dataframe with column information
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    st.write(col_info)
    
    # Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    findings = f"Dataset has {len(df)} rows and {len(df.columns)} columns. "
    findings += f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
    findings += f"and {len(datetime_cols)} datetime columns."
    
    add_summary(
        "Data Types", 
        "Analyzed column types and basic information",
        findings,
        "Check if data types are correctly assigned for each column."
    )
    
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols
    }

# Function to analyze missing values
def analyze_missing(df):
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Values': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    }).sort_values('Missing_Percentage', ascending=False)
    
    st.write(missing_data[missing_data['Missing_Values'] > 0])
    
    # Plot missing values
    if missing_data['Missing_Values'].sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data[missing_data['Missing_Values'] > 0].plot(
            x='Column', y='Missing_Percentage', kind='bar', ax=ax
        )
        ax.set_title('Missing Values Percentage by Column')
        ax.set_ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Summary
        high_missing = missing_data[missing_data['Missing_Percentage'] > 30]
        if len(high_missing) > 0:
            findings = f"Columns with high missing values (>30%): {', '.join(high_missing['Column'].tolist())}"
            recommendation = "Consider removing columns with high missing values or implement advanced imputation techniques."
        else:
            findings = "No columns have excessive missing values (>30%)."
            recommendation = "Use appropriate imputation methods for columns with missing values."
            
        add_summary(
            "Missing Values",
            "Analyzed pattern and percentage of missing values",
            findings,
            recommendation
        )
    else:
        st.write("No missing values found in the dataset.")
        add_summary(
            "Missing Values",
            "Analyzed pattern and percentage of missing values",
            "No missing values found in the dataset.",
            "No action needed for missing values."
        )

# Function to detect outliers
def detect_outliers(df, numeric_cols):
    st.subheader("Outlier Detection")
    
    if not numeric_cols:
        st.write("No numeric columns available for outlier detection.")
        return
    
    outlier_results = {}
    
    for col in numeric_cols:
        with st.expander(f"Outlier Analysis for {col}"):
            # Calculate outlier bounds using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            st.write(f"**Summary for {col}:**")
            st.write(f"- Lower bound: {lower_bound:.2f}")
            st.write(f"- Upper bound: {upper_bound:.2f}")
            st.write(f"- Number of outliers: {len(outliers)}")
            st.write(f"- Percentage of outliers: {outlier_percentage:.2f}%")
            
            # Box plot
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            df[col].plot(kind='box', ax=ax[0])
            ax[0].set_title(f'Box Plot of {col}')
            
            # Histogram with outlier bounds
            df[col].hist(bins=30, ax=ax[1])
            ax[1].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
            ax[1].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
            ax[1].set_title(f'Distribution of {col} with Outlier Bounds')
            ax[1].legend()
            
            st.pyplot(fig)
            
            outlier_results[col] = {
                'count': len(outliers),
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    # Summary
    high_outlier_cols = [col for col in outlier_results if outlier_results[col]['percentage'] > 5]
    if high_outlier_cols:
        findings = f"Columns with significant outliers (>5%): {', '.join(high_outlier_cols)}"
        recommendation = "Consider outlier treatment methods like capping, transformation, or removal for these columns."
    else:
        findings = "No columns have excessive outliers (>5%)."
        recommendation = "Outliers are within acceptable limits for all columns."
        
    add_summary(
        "Outlier Detection",
        "Used IQR method to detect outliers in numeric columns",
        findings,
        recommendation
    )
    
    return outlier_results

# Function for feature engineering suggestions
def suggest_feature_engineering(df, numeric_cols, categorical_cols, datetime_cols):
    st.subheader("Feature Engineering Suggestions")
    
    suggestions = []
    
    # Suggestions for numeric columns
    if numeric_cols:
        suggestions.append("**For numeric columns:**")
        suggestions.append("- Create interaction terms between related numeric variables")
        suggestions.append("- Apply transformations (log, square root) for skewed distributions")
        suggestions.append("- Create bins or categories from continuous variables")
        suggestions.append("- Calculate ratios between related variables")
    
    # Suggestions for categorical columns
    if categorical_cols:
        suggestions.append("**For categorical columns:**")
        suggestions.append("- Perform one-hot encoding for models that require numeric input")
        suggestions.append("- Create frequency encoding (replace categories with their frequency)")
        suggestions.append("- Target encoding for high cardinality categorical variables")
        suggestions.append("- Create interaction features between categorical variables")
    
    # Suggestions for datetime columns
    if datetime_cols:
        suggestions.append("**For datetime columns:**")
        suggestions.append("- Extract components: year, month, day, dayofweek, quarter, etc.")
        suggestions.append("- Create time-based features: is_weekend, is_holiday, season")
        suggestions.append("- Calculate time differences from a reference date")
        suggestions.append("- Create rolling statistics for time series data")
    
    # Display suggestions
    for suggestion in suggestions:
        st.write(suggestion)
    
    # Summary
    findings = f"Suggested {len(suggestions)} feature engineering ideas based on data types."
    recommendation = "Implement relevant feature engineering techniques to improve model performance."
    
    add_summary(
        "Feature Engineering",
        "Suggested feature engineering techniques based on data types",
        findings,
        recommendation
    )
    
    return suggestions

# Function for correlation analysis
def analyze_correlations(df, numeric_cols):
    st.subheader("Correlation Analysis")
    
    if len(numeric_cols) < 2:
        st.write("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Display correlation matrix
    st.write("Correlation Matrix:")
    st.write(corr_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        st.write("**Highly correlated pairs (|r| > 0.7):**")
        for pair in high_corr_pairs:
            st.write(f"- {pair[0]} and {pair[1]}: {pair[2]:.3f}")
        
        findings = f"Found {len(high_corr_pairs)} highly correlated variable pairs (|r| > 0.7)."
        recommendation = "Consider removing one variable from each highly correlated pair to avoid multicollinearity."
    else:
        st.write("No highly correlated pairs found (|r| > 0.7).")
        findings = "No highly correlated variable pairs found (|r| > 0.7)."
        recommendation = "No action needed for multicollinearity."
    
    add_summary(
        "Correlation Analysis",
        "Calculated Pearson correlations between numeric variables",
        findings,
        recommendation
    )
    
    return corr_matrix, high_corr_pairs

# Function for distribution analysis
def analyze_distributions(df, numeric_cols, categorical_cols):
    st.subheader("Distribution Analysis")
    
    # Numeric distributions
    if numeric_cols:
        st.write("### Numeric Variable Distributions")
        
        # Select a numeric column to analyze
        selected_num_col = st.selectbox("Select a numeric column to analyze:", numeric_cols)
        
        if selected_num_col:
            col_data = df[selected_num_col].dropna()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Histogram', 
                    'Box Plot',
                    'Q-Q Plot',
                    'Summary Statistics'
                ),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                      [{"type": "xy"}, {"type": "table"}]]
            )
            
            # Histogram
            fig.add_trace(go.Histogram(x=col_data, name='Histogram'), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=col_data, name='Box Plot'), row=1, col=2)
            
            # Q-Q plot
            qq = stats.probplot(col_data, dist="norm")
            x = np.array([qq[0][0][0], qq[0][0][-1]])
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q Plot'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines', name='Normal Fit'), row=2, col=1)
            
            # Summary statistics
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    len(col_data),
                    col_data.mean(),
                    col_data.std(),
                    col_data.min(),
                    col_data.quantile(0.25),
                    col_data.median(),
                    col_data.quantile(0.75),
                    col_data.max(),
                    col_data.skew(),
                    col_data.kurtosis()
                ]
            })
            stats_df['Value'] = stats_df['Value'].round(3)
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=[stats_df['Statistic'], stats_df['Value']])
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text=f"Distribution Analysis for {selected_num_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Normality test
            _, p_value = stats.normaltest(col_data)
            st.write(f"Normality test (p-value): {p_value:.4f}")
            if p_value < 0.05:
                st.write("p < 0.05 - distribution is significantly different from normal")
            else:
                st.write("p >= 0.05 - distribution is not significantly different from normal")
    
    # Categorical distributions
    if categorical_cols:
        st.write("### Categorical Variable Distributions")
        
        # Select a categorical column to analyze
        selected_cat_col = st.selectbox("Select a categorical column to analyze:", categorical_cols)
        
        if selected_cat_col:
            value_counts = df[selected_cat_col].value_counts()
            
            # Create bar chart
            fig = px.bar(
                x=value_counts.index.astype(str), 
                y=value_counts.values,
                labels={'x': selected_cat_col, 'y': 'Count'},
                title=f"Distribution of {selected_cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display value counts
            st.write("Value counts:")
            st.write(value_counts)
    
    # Summary
    findings = "Analyzed distributions of numeric and categorical variables."
    recommendation = "Consider transformations for highly skewed variables and encoding for categorical variables."
    
    add_summary(
        "Distribution Analysis",
        "Examined distributions of all variables",
        findings,
        recommendation
    )

# Function for time series analysis
def analyze_time_series(df, datetime_cols, numeric_cols):
    st.subheader("Time Series Analysis")
    
    if not datetime_cols:
        st.write("No datetime columns found for time series analysis.")
        return
    
    # Select datetime column
    date_col = st.selectbox("Select datetime column:", datetime_cols)
    
    if not date_col:
        return
    
    # Select numeric column for analysis
    value_col = st.selectbox("Select numeric column for time series analysis:", numeric_cols)
    
    if not value_col:
        return
    
    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set datetime index and sort
    ts_df = df.set_index(date_col).sort_index()
    
    # Plot time series
    fig = px.line(ts_df, y=value_col, title=f"Time Series of {value_col}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Decomposition
    st.write("### Time Series Decomposition")
    
    # Check if time series is regular
    if len(ts_df) < 2:
        st.write("Not enough data points for decomposition.")
        return
    
    # Handle missing values for decomposition
    ts_clean = ts_df[value_col].dropna()
    
    if len(ts_clean) < 50:
        st.write("Need at least 50 data points for reliable decomposition.")
        return
    
    # Determine frequency (try to infer)
    freq = pd.infer_freq(ts_clean.index)
    if freq is None:
        # If cannot infer, set a reasonable default
        if len(ts_clean) > 365:
            freq = 'D'  # daily
        else:
            freq = None
    
    try:
        decomposition = seasonal_decompose(ts_clean, model='additive', period=7 if freq is None else None)
        
        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary
        findings = "Performed time series decomposition to identify trend, seasonality, and residuals."
        recommendation = "Consider time series models that account for the identified patterns."
        
    except ValueError as e:
        st.write(f"Cannot decompose time series: {e}")
        findings = "Time series decomposition attempted but failed due to data characteristics."
        recommendation = "Check if the time series has sufficient data points and regular frequency."
    
    add_summary(
        "Time Series Analysis",
        "Analyzed time series patterns and decomposition",
        findings,
        recommendation
    )

# Function for group comparisons
def analyze_group_comparisons(df, numeric_cols, categorical_cols):
    st.subheader("Group Comparisons")
    
    if not categorical_cols or not numeric_cols:
        st.write("Need both categorical and numeric columns for group comparisons.")
        return
    
    # Select grouping variable
    group_col = st.selectbox("Select categorical variable for grouping:", categorical_cols)
    
    if not group_col:
        return
    
    # Select numeric variable to compare
    value_col = st.selectbox("Select numeric variable to compare across groups:", numeric_cols)
    
    if not value_col:
        return
    
    # Remove rows with missing values in selected columns
    comp_df = df[[group_col, value_col]].dropna()
    
    # Group statistics
    st.write("### Group Statistics")
    group_stats = comp_df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    st.write(group_stats)
    
    # Box plot by group
    fig = px.box(comp_df, x=group_col, y=value_col, title=f"{value_col} by {group_col}")
    st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA test if more than 2 groups
    groups = comp_df[group_col].unique()
    if len(groups) > 2:
        group_data = [comp_df[comp_df[group_col] == g][value_col] for g in groups]
        _, p_value = f_oneway(*group_data)
        
        st.write(f"**ANOVA test (p-value): {p_value:.4f}**")
        if p_value < 0.05:
            st.write("p < 0.05 - significant difference exists between groups")
        else:
            st.write("p >= 0.05 - no significant difference between groups")
        
        findings = f"ANOVA test showed {'significant' if p_value < 0.05 else 'no significant'} differences between groups."
        recommendation = "If significant differences exist, consider including this grouping variable in predictive models."
    else:
        # T-test for 2 groups
        if len(groups) == 2:
            group1_data = comp_df[comp_df[group_col] == groups[0]][value_col]
            group2_data = comp_df[comp_df[group_col] == groups[1]][value_col]
            _, p_value = stats.ttest_ind(group1_data, group2_data)
            
            st.write(f"**T-test (p-value): {p_value:.4f}**")
            if p_value < 0.05:
                st.write("p < 0.05 - significant difference exists between the two groups")
            else:
                st.write("p >= 0.05 - no significant difference between the two groups")
            
            findings = f"T-test showed {'significant' if p_value < 0.05 else 'no significant'} difference between the two groups."
            recommendation = "If significant difference exists, this grouping variable may be important for prediction."
        else:
            findings = "Only one group available for comparison."
            recommendation = "Need at least two groups for statistical comparison."
    
    add_summary(
        "Group Comparisons",
        "Compared numeric variable across categorical groups",
        findings,
        recommendation
    )

# Main app logic
if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Show basic info
    st.subheader("Basic Dataset Information")
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Analyze data types and get column categories
    col_categories = analyze_data_types(df)
    numeric_cols = col_categories['numeric_cols']
    categorical_cols = col_categories['categorical_cols']
    datetime_cols = col_categories['datetime_cols']
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Perform selected analyses
    if analysis_options['missing_values']:
        status_text.text("Analyzing missing values...")
        analyze_missing(df)
        progress_bar.progress(20)
    
    if analysis_options['outlier_detection'] and numeric_cols:
        status_text.text("Detecting outliers...")
        detect_outliers(df, numeric_cols)
        progress_bar.progress(40)
    
    if analysis_options['feature_engineering']:
        status_text.text("Generating feature engineering suggestions...")
        suggest_feature_engineering(df, numeric_cols, categorical_cols, datetime_cols)
        progress_bar.progress(60)
    
    if analysis_options['correlation_analysis'] and len(numeric_cols) >= 2:
        status_text.text("Analyzing correlations...")
        analyze_correlations(df, numeric_cols)
        progress_bar.progress(80)
    
    if analysis_options['distribution_analysis']:
        status_text.text("Analyzing distributions...")
        analyze_distributions(df, numeric_cols, categorical_cols)
        progress_bar.progress(90)
    
    if analysis_options['time_series_analysis'] and datetime_cols:
        status_text.text("Analyzing time series...")
        analyze_time_series(df, datetime_cols, numeric_cols)
        progress_bar.progress(95)
    
    if analysis_options['group_comparisons'] and categorical_cols and numeric_cols:
        status_text.text("Analyzing group comparisons...")
        analyze_group_comparisons(df, numeric_cols, categorical_cols)
        progress_bar.progress(100)
    
    status_text.text("Analysis complete!")
    
    # Display summaries
    st.subheader("Analysis Summaries")
    for i, summary in enumerate(st.session_state.summaries, 1):
        with st.expander(f"{i}. {summary['category']}: {summary['description']}"):
            st.write(f"**Findings:** {summary['findings']}")
            st.write(f"**Recommendation:** {summary['recommendation']}")
    
    # Download summary button
    if st.session_state.summaries:
        pdf_data = create_pdf(st.session_state.summaries)
        st.download_button(
            label="Download Summary Report (PDF)",
            data=pdf_data,
            file_name="eda_summary_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload a CSV file to begin analysis.")

# Example dataset and test run section
with st.expander("Click here to see an example with test data"):
    st.write("""
    ### Example Analysis with Sample Data
    
    Below is a sample dataset and the expected outputs from our EDA functions.
    """)
    
    # Create sample data
    np.random.seed(42)
    sample_size = 200
    
    sample_df = pd.DataFrame({
        'customer_id': range(1, sample_size + 1),
        'age': np.random.normal(45, 15, sample_size).astype(int),
        'income': np.random.lognormal(10, 1, sample_size),
        'credit_score': np.random.normal(650, 100, sample_size),
        'purchase_amount': np.random.exponential(100, sample_size),
        'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], sample_size),
        'membership_type': np.random.choice(['Basic', 'Premium', 'Gold'], sample_size, p=[0.6, 0.3, 0.1]),
        'signup_date': pd.date_range('2020-01-01', periods=sample_size, freq='D'),
        'last_purchase': pd.date_range('2023-01-01', periods=sample_size, freq='D')
    })
    
    # Add some missing values
    sample_df.loc[sample_df.sample(10).index, 'age'] = np.nan
    sample_df.loc[sample_df.sample(5).index, 'income'] = np.nan
    sample_df.loc[sample_df.sample(15).index, 'credit_score'] = np.nan
    
    # Add some outliers
    sample_df.loc[sample_df.sample(3).index, 'income'] = sample_df['income'].max() * 3
    sample_df.loc[sample_df.sample(2).index, 'purchase_amount'] = sample_df['purchase_amount'].max() * 5
    
    st.write("**Sample Dataset:**")
    st.write(sample_df.head())
    
    st.write("**Testing Analysis Functions:**")
    
    # Test data type analysis
    st.write("1. Data Type Analysis:")
    col_categories = analyze_data_types(sample_df)
    st.write(f"Numeric columns: {col_categories['numeric_cols']}")
    st.write(f"Categorical columns: {col_categories['categorical_cols']}")
    st.write(f"Datetime columns: {col_categories['datetime_cols']}")
    
    # Test missing value analysis
    st.write("2. Missing Value Analysis:")
    analyze_missing(sample_df)
    
    # Test outlier detection
    st.write("3. Outlier Detection:")
    outlier_results = detect_outliers(sample_df, col_categories['numeric_cols'])
    
    st.write("This example demonstrates how the EDA app would analyze a sample dataset.")
