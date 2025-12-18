import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.colors
import pickle
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Depression Analysis Dashboard",
    page_icon="📊"
)

# --- Global Constants and Configurations ---
DATA_FILE_PATH = "data_with_predictions.xlsx"
MODEL_FILE_PATH = "depression_xgmodel.pkl"

DEPRESSION_COLORS = {
    'Yes': '#FF6347',
    'No': '#4682B4'
}
PREDICTION_STATUS_COLORS = {
    1: '#FF6347',
    0: '#4682B4'
}
ACTUAL_PREDICTED_BAR_COLORS = {
    'Actual': 'gray',
    'Predicted': 'teal'
}

MODEL_INPUT_FEATURES = [
    'Age', 'Work Hours', 'Work Pressure', 'Job Satisfaction',
    'Have you ever had suicidal thoughts ?', 'Financial Stress', 'Family History of Mental Illness'
]

NUMERICAL_FEATURES_TO_SCALE = [
    'Age', 'Work Pressure', 'Job Satisfaction', 'Work Hours', 'Financial Stress'
]

CATEGORICAL_FEATURES_TO_ENCODE = [
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
]

# --- Data and Model Loading Functions ---

@st.cache_data
def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df_loaded = pd.read_excel(file_path)
        return df_loaded
    except FileNotFoundError:
        st.error(f"Data file not found: '{file_path}'. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the data file '{file_path}': {e}")
        st.stop()

@st.cache_resource
def load_model(file_path):
    """Loads the trained machine learning model."""
    try:
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except FileNotFoundError:
        st.error(f"Model file not found: '{file_path}'. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model from '{file_path}': {e}")
        st.stop()

# Load Data and Model
df = load_data(DATA_FILE_PATH)
model = load_model(MODEL_FILE_PATH)

# Initialize and Fit Preprocessors Globally
scaler = StandardScaler()
scaler.fit(df[NUMERICAL_FEATURES_TO_SCALE])

label_encoders = {}
for col in CATEGORICAL_FEATURES_TO_ENCODE:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    label_encoders[col] = le

# --- Preprocessing Helper Function for What-If Analysis Inputs ---

def _preprocess_input_for_prediction(raw_input_df):
    """
    Preprocesses raw input data for model prediction.
    """
    processed_df = raw_input_df.copy()

    for col in CATEGORICAL_FEATURES_TO_ENCODE:
        if col in processed_df.columns:
            processed_df[col] = label_encoders[col].transform(processed_df[col].astype(str))
        else:
            processed_df[col] = 0

    if not processed_df[NUMERICAL_FEATURES_TO_SCALE].empty:
        processed_df[NUMERICAL_FEATURES_TO_SCALE] = scaler.transform(processed_df[NUMERICAL_FEATURES_TO_SCALE])
    
    aligned_df = processed_df[MODEL_INPUT_FEATURES].copy()
    
    return aligned_df

# Derive the exact column order the model expects after preprocessing
dummy_input_data = {feature: df[feature].iloc[0] for feature in MODEL_INPUT_FEATURES}
dummy_df = pd.DataFrame([dummy_input_data])
MODEL_EXPECTED_COLUMN_ORDER = _preprocess_input_for_prediction(dummy_df).columns.tolist()

# --- Sidebar for Filtering ---
st.sidebar.header("📊 Filter Data")

with st.sidebar:
    gender_filter = st.multiselect(
        "Gender",
        options=df['Gender'].unique(),
        help="Filter data by gender."
    )
    suicidal_thoughts_filter = st.multiselect(
        "Suicidal thoughts",
        options=df['Have you ever had suicidal thoughts ?'].unique(),
        help="Filter by history of suicidal thoughts."
    )
    age_range_filter = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max())),
        help="Filter data by age range."
    )
    work_pressure_filter = st.multiselect(
        "Work Pressure",
        options=sorted(df['Work Pressure'].unique()),
        help="Filter data by work pressure level (1-5)."
    )
    financial_stress_filter = st.multiselect(
        "Financial Stress",
        options=sorted(df['Financial Stress'].unique()),
        help="Filter data by financial stress level (1-5)."
    )
    high_risk_threshold = st.slider(
        "Select High Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the threshold to define 'High Risk' based on the predicted depression risk score."
    )

# Apply filters
filtered_df = df.copy()

if gender_filter:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
if suicidal_thoughts_filter:
    filtered_df = filtered_df[filtered_df['Have you ever had suicidal thoughts ?'].isin(suicidal_thoughts_filter)]
if work_pressure_filter:
    filtered_df = filtered_df[filtered_df['Work Pressure'].isin(work_pressure_filter)]
if financial_stress_filter:
    filtered_df = filtered_df[filtered_df['Financial Stress'].isin(financial_stress_filter)]

filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range_filter[0]) &
    (filtered_df['Age'] <= age_range_filter[1])
]

# --- Main Dashboard Content ---

st.title("🧠 Depression Analysis Dashboard")

# KPI Summary Section
#st.subheader("KPI Summary on Filtered Data")
if filtered_df.empty:
    st.warning("No data matches the selected filters for KPI summary.")
else:
    total_records = filtered_df.shape[0]
    total_actual_depression_cases = filtered_df['Depression'].value_counts().get('Yes', 0)
    average_predicted_risk = filtered_df['Predicted_Depression_Risk'].mean()
    high_risk_count = filtered_df[filtered_df['Predicted_Depression_Risk'] > high_risk_threshold].shape[0]

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        with st.container(border=True):
            st.metric("📋 Total Individual Records", total_records)
    with col_kpi2:
        with st.container(border=True):
            st.metric("❗ Total Actual Depression Cases", total_actual_depression_cases)
    with col_kpi3:
        with st.container(border=True):
            st.metric("📊 Average Predicted Depression Risk", f"{average_predicted_risk:.2f}")
    with col_kpi4:
        with st.container(border=True):
            st.metric(f"⚠️ High Risk Depression Count (> {high_risk_threshold:.2f})", high_risk_count)

# --- Visualizations ---
with st.container():
    col1_vis, col2_vis, col3_vis = st.columns([1, 2, 1])
    key_style = "font-size: 0.85em; color: #555555;"
    with col1_vis:
        if filtered_df.empty:
            st.warning("No data available for EDA visualizations with the current filters.")
        else:
            with st.container(border=True):
                #st.write("### Distribution of Depression Status")
                depression_counts = filtered_df['Depression'].value_counts().reset_index()
                depression_counts.columns = ['Depression', 'Count']
                fig_dep_dist = px.pie(
                    depression_counts,
                    names='Depression',
                    values='Count',
                    title='Distribution of Depression Status',
                    color='Depression',
                    color_discrete_map=DEPRESSION_COLORS
                )
                fig_dep_dist.update_traces(textposition='inside', textinfo='percent+label')
                fig_dep_dist.update_layout(showlegend=True, margin={"t":50, "b":0, "l":0, "r":0}, legend_title_text="Depression")
                st.plotly_chart(fig_dep_dist, use_container_width=True)

                 # --- Dynamic Key Notes/Findings for Distribution of Depression Status ---
                total_records_pie = len(filtered_df)
                non_depressed_pie_count = depression_counts.loc[depression_counts['Depression'] == 'No', 'Count'].sum()
                depressed_pie_count = depression_counts.loc[depression_counts['Depression'] == 'Yes', 'Count'].sum()

                if total_records_pie > 0:
                    non_depressed_pie_percent = (non_depressed_pie_count / total_records_pie) * 100
                    depressed_pie_percent = (depressed_pie_count / total_records_pie) * 100
                else:
                    non_depressed_pie_percent = 0
                    depressed_pie_percent = 0

                st.markdown(f"<span style='{key_style}'> **Conclusion:** In this view, **{non_depressed_pie_count}** are not depressed, and **{depressed_pie_count}** are depressed.</span>", unsafe_allow_html=True)

            with st.container(border=True):
                # --- Define the binning manually ---
                min_age = filtered_df['Age'].min()
                max_age = filtered_df['Age'].max()
                nbins = 6
                bin_edges = np.linspace(min_age, max_age, nbins + 1) 
                
                # Create bin labels like '20–29', '30–39', ...
                bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])-1}" for i in range(len(bin_edges)-1)]

                # --- Create histogram with fixed bins ---
                # Assign bin labels to each row using pd.cut
                filtered_df['AgeGroup'] = pd.cut(
                    filtered_df['Age'],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True,
                    right=False  # So that 20 is included in 20–29
                )

                # --- Plot chart using the AgeGroup column ---
                fig_age_dist = px.histogram(
                    filtered_df,
                    x='AgeGroup',
                    title='Distribution of Age by Depression',
                    color='Depression',
                    color_discrete_map=DEPRESSION_COLORS,
                    barmode='group',
                    category_orders={'AgeGroup': bin_labels}  # ensures correct bin order
                )

                fig_age_dist.update_layout(
                    xaxis_title='Age Group',
                    yaxis_title='Frequency',
                    bargap=0.1,
                    margin={"t": 50, "b": 20, "l": 10, "r": 10}
                )
                st.plotly_chart(fig_age_dist, use_container_width=True)

                # Generate Key Notes
                conclusion_age = ""

                if 'Depression' in filtered_df.columns and not filtered_df.empty:
                    depressed_ages_df = filtered_df[filtered_df['Depression'] == 'Yes']

                    if not depressed_ages_df.empty:
                        depressed_age_counts = depressed_ages_df['AgeGroup'].value_counts().sort_index()

                        if not depressed_age_counts.empty:
                            peak_group = depressed_age_counts.idxmax()
                            peak_count = depressed_age_counts.max()

                            conclusion_age = (
                                f"Most depression cases were found among people aged **{peak_group}**, "
                                f"with a total of **{int(peak_count)}** individuals in that group."
                            )
                        else:
                            conclusion_age = "There is no clear age group where depression is most common."
                    else:
                        conclusion_age = "There are no individuals with depression in the current filtered data."
                else:
                    conclusion_age = "Age or depression information is missing from the data."

                # --- Display ---
                st.markdown(f"<span style='{key_style}'>**Conclusion:** {conclusion_age}</span>", unsafe_allow_html=True)
                                
    with col2_vis:
        if filtered_df.empty:
            st.warning("No data available for prediction analysis visualizations with the current filters.")
        else:
            with st.container(border=True):
                #st.write("### Risk Factor and Correlation")

                # Encode binary categorical feature
                temp_df_corr = filtered_df.copy()
                temp_df_corr['Depression_Binary'] = temp_df_corr['Depression'].map({'Yes': 1, 'No': 0})
                temp_df_corr['Suicidal_Thoughts_Binary'] = temp_df_corr['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
                temp_df_corr['Family_History_Mental_Illness'] = temp_df_corr['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

                # Encode Ordinal Categorical Feature
                # Sleep Duration: Assign numerical values based on typical understanding of duration
                sleep_duration_order = {
                    'Less than 5 hours': 1,
                    '5-6 hours': 2,
                    '7-8 hours': 3,
                    'More than 8 hours': 4
                }
                temp_df_corr['Sleep_Duration'] = temp_df_corr['Sleep Duration'].map(sleep_duration_order)

                # Dietary Habits: Assign numerical values based on healthiness
                dietary_habits_order = {
                    'Unhealthy': 1,
                    'Moderate': 2,
                    'Healthy': 3
                }
                temp_df_corr['Dietary_Habits'] = temp_df_corr['Dietary Habits'].map(dietary_habits_order)

                numerical_cols_for_corr = temp_df_corr.select_dtypes(include=np.number).columns.tolist()
                cols_to_exclude_corr = ['Gender','Predicted_Depression', 'Predicted_Depression_Risk',
                                         'Predicted_Depression_Status', 'Unnamed: 0']
                
                correlation_features = [col for col in numerical_cols_for_corr if col not in cols_to_exclude_corr]
                if 'Depression_Binary' not in correlation_features:
                    correlation_features.append('Depression_Binary')

                corr_matrix = temp_df_corr[correlation_features].corr()

                fig_corr_matrix = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    title = 'Risk Factor and Correlation'
                    #title='Correlation Matrix of Numerical Features and Depression'
                )
                fig_corr_matrix.update_layout(margin={"t":50, "b":0, "l":0, "r":0})
                st.plotly_chart(fig_corr_matrix, use_container_width=True)
                #st.markdown() #put overview of charting
                # --- Dynamic Key Notes/Findings for Correlation Heatmap (Simple) ---
                conclusion_corr = ""
                if 'Depression_Binary' in corr_matrix.columns:
                    depression_correlations = corr_matrix['Depression_Binary'].drop('Depression_Binary').sort_values(ascending=False)

                    # Identify strongest positive and negative correlations (above threshold)
                    strongest_positive = depression_correlations[depression_correlations > 0.1].head(1)
                    strongest_negative = depression_correlations[depression_correlations < -0.1].tail(1)

                    pos_text = ""
                    neg_text = ""

                    if not strongest_positive.empty:
                        pos_feature = strongest_positive.index[0].replace('_Binary', '').replace('_Encoded', '').replace('_', ' ').title()
                        pos_text = f"People with higher **{pos_feature.lower()}** tend to show **more signs of depression**"

                    if not strongest_negative.empty:
                        neg_feature = strongest_negative.index[0].replace('_Binary', '').replace('_Encoded', '').replace('_', ' ').title()
                        neg_text = f"People with higher **{neg_feature.lower()}** tend to show **fewer signs of depression**"

                    if pos_text and neg_text:
                        conclusion_corr = f"{pos_text}, while {neg_text}."
                    elif pos_text:
                        conclusion_corr = pos_text + "."
                    elif neg_text:
                        conclusion_corr = neg_text + "."
                    else:
                        conclusion_corr = "No clear relationship between any factor and depression was found."
                else:
                    conclusion_corr = "Depression data is not available for this analysis."

                # --- Display ---
                st.markdown(f"<span style='{key_style}'>**Conclusion:** {conclusion_corr}</span>", unsafe_allow_html=True)
                

            with st.container(border=True):
                 # Aggregate data to get the mean Job Satisfaction for each Work Pressure and Depression group
                if not filtered_df.empty:
                    agg_df = filtered_df.groupby(['Work Pressure', 'Depression'])['Job Satisfaction'].mean().reset_index()

                    if not agg_df.empty:
                        fig_line_js_wp = px.line(
                            agg_df,
                            x='Work Pressure',
                            y='Job Satisfaction',
                            color='Depression',
                            title='Average Job Satisfaction by Work Pressure and Depression',
                            color_discrete_map=DEPRESSION_COLORS,
                            markers=True, # Show markers at each data point
                            labels={
                                "Work Pressure": "Work Pressure Level",
                                "Job Satisfaction": "Average Job Satisfaction Level"
                            }
                        )

                        fig_line_js_wp.update_layout(
                            xaxis_title='Work Pressure Level',
                            yaxis_title='Average Job Satisfaction Level',
                            margin={"t":50, "b":0, "l":0, "r":0},
                            legend_title_text="Depression"
                        )
                        fig_line_js_wp.update_xaxes(type='category') # Treat Work Pressure as categorical for discrete points
                        st.plotly_chart(fig_line_js_wp, use_container_width=True)

                        # --- Dynamic Key Notes/Findings for Line Chart ---
                        conclusion_line = ""
                        if not agg_df.empty:
                            highest_wp_level = agg_df['Work Pressure'].max()

                            # Filter depression groups
                            depressed_line_data = agg_df[agg_df['Depression'] == 'Yes']
                            non_depressed_line_data = agg_df[agg_df['Depression'] == 'No']

                            # Get average job satisfaction values
                            highest_wp_dep_js = depressed_line_data[
                                depressed_line_data['Work Pressure'] == highest_wp_level
                            ]['Job Satisfaction'].iloc[0] if not depressed_line_data.empty else None

                            highest_wp_non_dep_js = non_depressed_line_data[
                                non_depressed_line_data['Work Pressure'] == highest_wp_level
                            ]['Job Satisfaction'].iloc[0] if not non_depressed_line_data.empty else None

                            # Get sample sizes
                            dep_n = len(filtered_df[
                                (filtered_df['Work Pressure'] == highest_wp_level) &
                                (filtered_df['Depression'] == 'Yes')
                            ])
                            non_dep_n = len(filtered_df[
                                (filtered_df['Work Pressure'] == highest_wp_level) &
                                (filtered_df['Depression'] == 'No')
                            ])

                            # Build the simplified conclusion
                            if highest_wp_dep_js is not None and highest_wp_non_dep_js is not None:
                                conclusion_line = (
                                    f"At the highest work pressure level (**{highest_wp_level}**), people with depression report a lower "
                                    f"average job satisfaction with **{highest_wp_dep_js:.1f}** ({dep_n} individuals) compared to those without depression "
                                    f"with **{highest_wp_non_dep_js:.1f}** ({non_dep_n} individuals), showing a clear difference between the two groups."
                                )
                            else:
                                conclusion_line = "Not enough data to compare job satisfaction between groups at the highest work pressure level."

                        # --- Display ---
                        st.markdown(f"<span style='{key_style}'>**Conclusion:** {conclusion_line}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No aggregated data available to plot the line chart with current filters.")
                else:
                    st.info("No data available for line chart with the current filters.")

    with col3_vis:
        if filtered_df.empty:
            st.warning("No data available for visualization with the current filters.")
        else:
            with st.container(border=True):
                #st.subheader("Work Hours Distribution by Work Pressure and Depression Status")

                # Create a bar chart to show the count of depression status per gender
                fig_gender_depression = px.histogram(
                    filtered_df,
                    x='Gender',
                    color='Depression',
                    barmode='stack', # Use 'group' to show bars side-by-side, 'stack' for stacked bars
                    title='Distribution of Depression by Gender',
                    color_discrete_map=DEPRESSION_COLORS,
                    labels={
                        "Gender": "Gender",
                        "count": "Number of Individuals"
                    },
                    category_orders={"Depression": ["No", "Yes"]} # Ensure 'No' comes before 'Yes' in legend/bars
                )

                # Update layout for better aesthetics
                fig_gender_depression.update_layout(
                    margin={"t":50, "b":0, "l":0, "r":0},
                    xaxis_title="Gender",
                    yaxis_title="Number of Individuals",
                    legend_title="Depression"
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig_gender_depression, use_container_width=True)

                # --- Dynamic Key Notes/Findings for Depression by Gender (Concise) ---
                conclusion_gender = ""
                total_records_gender = len(filtered_df)

                if total_records_gender > 0:
                    gender_depression_pivot = (
                        filtered_df.groupby('Gender')['Depression']
                        .value_counts()
                        .unstack()
                        .fillna(0)
                    )

                    if 'Yes' in gender_depression_pivot.columns and not gender_depression_pivot.empty:
                        highest_dep_gender_count = gender_depression_pivot['Yes'].max()
                        highest_dep_gender_name = gender_depression_pivot['Yes'].idxmax()

                        lowest_dep_gender_count = gender_depression_pivot['Yes'].min()
                        lowest_dep_gender_name = gender_depression_pivot['Yes'].idxmin()

                        if highest_dep_gender_name != lowest_dep_gender_name:
                            conclusion_gender = (
                                f"Depression is more common among **{highest_dep_gender_name.lower()}s** "
                                f"with **{int(highest_dep_gender_count)}** reported cases, compared to "
                                f"**{int(lowest_dep_gender_count)}** among **{lowest_dep_gender_name.lower()}s**."
                            )
                        else:
                            conclusion_gender = "Depression cases appear to be similar across genders in this dataset."
                    else:
                        conclusion_gender = "There are no recorded depression cases available for gender comparison."
                else:
                    conclusion_gender = "No gender or depression data available for analysis."

                # --- Display ---
                st.markdown(f"<span style='{key_style}'>{conclusion_gender}</span>", unsafe_allow_html=True)

            with st.container(border=True):
                #st.subheader("Suicidal Thoughts vs. Depression Status")
                suicidal_depression_counts = pd.crosstab(filtered_df['Have you ever had suicidal thoughts ?'], filtered_df['Depression']).reset_index()
                suicidal_depression_melted = suicidal_depression_counts.melt(id_vars=['Have you ever had suicidal thoughts ?'], var_name='Depression Status', value_name='Count')
                
                fig_suicidal_thoughts = px.bar(
                    suicidal_depression_melted,
                    x='Have you ever had suicidal thoughts ?',
                    y='Count',
                    color='Depression Status',
                    barmode='stack',
                    title='Suicidal Thoughts vs. Depression',
                    color_discrete_map=DEPRESSION_COLORS
                )
                fig_suicidal_thoughts.update_layout(xaxis_title='Have you ever had suicidal thoughts?', yaxis_title='Number of Individuals', margin={"t":50, "b":0, "l":0, "r":0})
                st.plotly_chart(fig_suicidal_thoughts, use_container_width=True)
                # --- Dynamic Key Notes/Findings for Suicidal Thoughts vs. Depression (Concise) ---
                conclusion_suicidal = ""
                if 'Yes' in suicidal_depression_counts.columns and not filtered_df.empty:
                    # Get 'Yes' suicidal thoughts row
                    suicidal_yes_row = suicidal_depression_counts[
                        suicidal_depression_counts['Have you ever had suicidal thoughts ?'] == 'Yes'
                    ]
                    suicidal_no_row = suicidal_depression_counts[
                        suicidal_depression_counts['Have you ever had suicidal thoughts ?'] == 'No'
                    ]

                    suicidal_yes_depressed = suicidal_yes_row['Yes'].sum() if not suicidal_yes_row.empty else 0
                    suicidal_yes_total = suicidal_yes_row['Yes'].sum() + suicidal_yes_row['No'].sum() if not suicidal_yes_row.empty else 0

                    suicidal_no_depressed = suicidal_no_row['Yes'].sum() if not suicidal_no_row.empty else 0
                    suicidal_no_total = suicidal_no_row['Yes'].sum() + suicidal_no_row['No'].sum() if not suicidal_no_row.empty else 0

                    if suicidal_yes_total > 0:
                        percent_yes_dep = (suicidal_yes_depressed / suicidal_yes_total) * 100
                        conclusion_suicidal = (
                            f"Among those who had suicidal thoughts, **{suicidal_yes_depressed}** were depressed"
                        )

                        if suicidal_no_depressed > 0:
                            conclusion_suicidal += (
                                f" — but even among those who didn’t, **{suicidal_no_depressed}** still showed signs of depression."
                            )
                        else:
                            conclusion_suicidal += ". No depression was found among those without suicidal thoughts."
                    elif suicidal_no_total > 0:
                        conclusion_suicidal = (
                            f"There were no individuals with suicidal thoughts in this view, "
                            f"but **{suicidal_no_depressed}** people without suicidal thoughts were still depressed."
                        )
                    else:
                        conclusion_suicidal = "No valid data for suicidal thoughts or depression in this view."
                else:
                    conclusion_suicidal = "Suicidal thought or depression data is not available for analysis."

                # --- Display ---
                st.markdown(f"<span style='{key_style}'>**Conclusion:**{conclusion_suicidal}</span>", unsafe_allow_html=True)

# --- Simulate a Scenario Section (Moved to a new row) ---
st.markdown("---") # Add a separator for clarity
st.subheader("🔬 Simulate a Scenario")

# Define ranges and options based on the full original dataframe for consistency
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
work_pressure_min, work_pressure_max = int(df['Work Pressure'].min()), int(df['Work Pressure'].max())
work_hours_min, work_hours_max = int(df['Work Hours'].min()), int(df['Work Hours'].max())
financial_stress_min, financial_stress_max = int(df['Financial Stress'].min()), int(df['Financial Stress'].max())
job_satisfaction_min, job_satisfaction_max = int(df['Job Satisfaction'].min()), int(df['Job Satisfaction'].max())

suicidal_thoughts_options = df['Have you ever had suicidal thoughts ?'].dropna().unique().tolist()
family_illness_options = df['Family History of Mental Illness'].dropna().unique().tolist()

with st.form("what_if_form"):
    st.write("Adjust parameters to see the predicted depression risk:")
    # Input widgets for the 6 features
    wi_age = st.slider("Age", age_min, age_max, 30, help=f"Select the individual's age (between {age_min} and {age_max}).")
    wi_work_pressure = st.slider("Work Pressure", work_pressure_min, work_pressure_max, 2, help=f"Select the work pressure level (between {work_pressure_min} and {work_pressure_max}).")
    wi_work_hours = st.slider("Work Hours", work_hours_min, work_hours_max, 8, help=f"Select the number of work hours per week (between {work_hours_min} and {work_hours_max}).")
    wi_job_satisfaction = st.slider("Job Satisfaction", job_satisfaction_min, job_satisfaction_max, 3, help=f"Select the job satisfaction level (between {job_satisfaction_min} and {job_satisfaction_max}).")
    wi_suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", suicidal_thoughts_options, help="Indicate if there is a history of suicidal thoughts.")
    wi_financial_stress = st.slider("Financial Stress", financial_stress_min, financial_stress_max, 3, help=f"Select the financial stress level (between {financial_stress_min} and {financial_stress_max}).")
    wi_family = st.selectbox("Family History of Mental Illess", family_illness_options, help="Indicate if there any family history.")

    predict_button = st.form_submit_button("Predict Depression Risk")

if predict_button:
    input_data_raw = {
        'Age': wi_age,
        'Work Hours': wi_work_hours,
        'Work Pressure': wi_work_pressure,
        'Job Satisfaction': wi_job_satisfaction,
        'Have you ever had suicidal thoughts ?': wi_suicidal_thoughts,
        'Financial Stress': wi_financial_stress,
        'Family History of Mental Illness': wi_family,
    }
    input_df_raw = pd.DataFrame([input_data_raw])

    try:
        input_df_processed_for_model = _preprocess_input_for_prediction(input_df_raw)

        predicted_risk = model.predict_proba(input_df_processed_for_model)[:, 1][0]
        predicted_status = "At Risk of Depression" if predicted_risk > 0.6 else "No Depression"

        st.subheader("Prediction Result")
        st.info(f"**Predicted Depression Risk:** {predicted_risk:.2%}")
        st.info(f"**Predicted Status:** {predicted_status}")

        
        # --- Dynamic Alert/Recommendation Logic ---
        alert_messages = []

        # High Financial Stress & Suicidal Thoughts
        if wi_financial_stress >= 4 and wi_suicidal_thoughts == 'Yes':
            alert_messages.append("⚠️ **High Concern:** High financial stress combined with a history of suicidal thoughts. Please seek immediate professional help.")

        # Very High Work Pressure & Low Job Satisfaction
        if wi_work_pressure >= 4 and wi_job_satisfaction <= 2:
            alert_messages.append("❗ **Work-Related Stress:** High work pressure and low job satisfaction can significantly impact mental health. Consider stress management techniques or seeking workplace support.")

        # Any Suicidal Thoughts (only show this if not already included above)
        if wi_suicidal_thoughts == 'Yes':
            if not any("High Concern" in msg for msg in alert_messages):
                alert_messages.append("🚨 **Immediate Attention:** A history of suicidal thoughts is a serious concern. Please reach out to a mental health professional or a crisis hotline immediately.")

        # High Financial Stress (standalone if not combined with suicidal thoughts)
        if wi_financial_stress >= 4 and not (wi_financial_stress >= 4 and wi_suicidal_thoughts == 'Yes'):
            alert_messages.append("📈 **Financial Strain:** High financial stress can contribute to mental health problems. Consider seeking financial counseling or stress reduction strategies.")

        # High Work Pressure (standalone)
        if wi_work_pressure >= 4 and not (wi_work_pressure >= 4 and wi_job_satisfaction <= 2):
            alert_messages.append("😖 **High Work Pressure:** Sustained high work pressure can be detrimental. Explore coping mechanisms or discuss workload with your employer.")

        # Very High Work Hours
        if wi_work_hours >= 50: # Example threshold for very high work hours
            alert_messages.append("⏳ **Excessive Work Hours:** Working long hours can lead to burnout. Ensure you are taking adequate breaks and maintaining work-life balance.")

        # Family History of Mental Illness
        if wi_family == 'Yes':
            alert_messages.append("🧬 **Family History:** A family history of mental illness can increase risk. Be proactive about your mental health and consider regular check-ups.")

        # Display alerts
        if alert_messages:
            st.markdown("---")
            st.subheader("Recommendations & Alerts")
            for msg in alert_messages:
                if "⚠️" in msg or "🚨" in msg: # Use st.error for critical alerts
                    st.error(msg)
                else: # Use st.warning for general recommendations
                    st.warning(msg)
            st.markdown("---")
            st.info("For any mental health concerns, please consult a qualified healthcare professional.")


    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Please ensure the input values are valid and the model's expected features are correctly generated. If the error persists, verify that your 'depression_xgmodel.pkl' was indeed trained on *exactly* these features and that the preprocessing steps in the Streamlit app exactly match your training script.")
