import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Generate QR code for the app
import qrcode
from PIL import Image
import io
import base64
import os
import re

# Set page config
# st.set_page_config(
#     page_title="Synapse Annotation Dashboard",
#     page_icon="🧠",
#     layout="wide"
# )

# Add logo and title in a row
col1, col2 = st.columns([1, 4])

with col1:
    # Add your logo - replace with your actual logo path
    st.image("./data/logo/catena_logo.png", width=120)

with col2:
    st.title(" Synapse Proofreading Dashboard")
    st.markdown(
        "<p style='font-size: 14px; color: gray;'>© 2025, Samia Mohinta, University of Cambridge. All rights reserved.</p>",
        unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Octo False Positives":
        df = pd.read_csv(
            './data/final_df_postsyn_transaction_octo_merged_ac.csv')
    elif dataset_name == "MR143 False Positives":  # Add more datasets as needed
        df = pd.read_csv(
            './data/final_df_postsyn_transaction_mr143.csv')

    # Convert execution_time to datetime
    df['execution_time'] = pd.to_datetime(df['execution_time'])

    # Add dataset identifier column
    df['dataset'] = dataset_name
    return df


# Dataset selection in sidebar
st.sidebar.header("Dataset Selection")
dataset_options = ["Octo False Positives", "MR143 False Positives"]  # Add more options as needed
selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)

# Load selected dataset
df = load_data(selected_dataset)

# Update cube creation to include dataset identifier
df['cube_ds_pos'] = df.apply(
    lambda x: f"{x['dataset']}_({int(x['post_x'] // 100)}, {int(x['post_y'] // 100)}, {int(x['post_z'] // 100)})",
    axis=1)

# Create simplified cube numbers
print(df['cube'].unique())
df['cube_number'] = df['cube'].apply(lambda x: f"cube#{x[-1]}")

# Sidebar filters
st.sidebar.header("Filters")
selected_users = st.sidebar.multiselect(
    "Select Users",
    options=df['user'].unique(),
    default=df['user'].unique()
)

# Main dashboard
# st.title("🧠 Synapse Annotation Analysis Dashboard")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Leaderboard", "Cube Analysis", "User Agreement", "Time Analysis", "Statistics", "Media Examples"])

# Add QR code and copyright in the sidebar footer
st.sidebar.markdown("---")
st.sidebar.subheader("Access on Mobile")


def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert PIL image to bytes
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


def plot_user_agreement(df, selected_users):
    """
    Create a publication-quality plot showing how many users agree with the benchmark annotations.
    Benchmark annotations are from user 'ac' with values 'correct', 'incorrect', or 'dubious'.
    Other users' annotations are mapped to these benchmark categories.
    """
    # Create a copy of the dataframe to avoid modifying the original
    agreement_df = df.copy()

    # Extract the user initials and annotation type from other_annotations column
    agreement_df['user_initial'] = agreement_df['other_annotations'].str.split(':').str[0].str.strip()
    agreement_df['annotation_type'] = agreement_df['other_annotations'].str.split(':').str[1].str.strip()

    # Map other users' annotations to benchmark categories
    mapping = {
        'pre correct': 'correct',
        'uncertain': 'dubious',
        # 'pushed false positives synapses': 'incorrect',
        # 'pushed to synapse': 'correct',
        'distanced set': 'incorrect',
        'wrong set': 'incorrect',
        # Add more mappings if needed
        'correct': 'correct',  # For benchmark user
        'incorrect': 'incorrect',  # For benchmark user
        'dubious': 'dubious'  # For benchmark user
    }

    # Debug: Check for missing mappings
    missing_mappings = [at for at in agreement_df['annotation_type'].unique() if at not in mapping]
    if missing_mappings:
        st.write("WARNING: The following annotation types have no mapping:")
        st.write(missing_mappings)
        # Add missing mappings with a default value
        for missing in missing_mappings:
            if 'correct' in missing.lower():
                mapping[missing] = 'correct'
            elif 'wrong' in missing.lower() or 'incorrect' in missing.lower():
                mapping[missing] = 'incorrect'
            elif 'uncertain' in missing.lower() or 'dubious' in missing.lower():
                mapping[missing] = 'dubious'
            else:
                mapping[missing] = 'unknown'
        st.write("Updated mapping:")
        st.write(mapping)

    # Apply mapping to standardize annotation types
    agreement_df['standardized_annotation'] = agreement_df['annotation_type'].map(mapping)

    # Find benchmark annotations (from user 'ac')
    benchmark_annotations = agreement_df[agreement_df['user_initial'] == 'ac'].copy()
    benchmark_annotations['benchmark_annotation'] = benchmark_annotations['standardized_annotation']

    # Keep only the necessary columns from benchmark
    benchmark_columns = ['post_x', 'post_y', 'post_z', 'benchmark_annotation']
    benchmark_subset = benchmark_annotations[benchmark_columns]

    # Merge benchmark annotations with all annotations
    merged_df = pd.merge(
        agreement_df,
        benchmark_subset,
        on=['post_x', 'post_y', 'post_z'],
        how='inner'
    )

    # Calculate agreement
    merged_df['agrees_with_benchmark'] = (merged_df['standardized_annotation'] == merged_df['benchmark_annotation'])

    # Group by user and benchmark annotation type
    agreement_stats = merged_df.groupby(['user', 'benchmark_annotation']).agg({
        'agrees_with_benchmark': ['count', 'sum']
    }).reset_index()

    # Flatten the column hierarchy
    agreement_stats.columns = ['User', 'Benchmark Annotation', 'Total Annotations', 'Agreements']

    # Calculate agreement percentage
    agreement_stats['Agreement Percentage'] = (
            agreement_stats['Agreements'] / agreement_stats['Total Annotations'] * 100).round(1)

    return agreement_stats


# Function to create a heatmap for agreement visualization
def plot_agreement_heatmap(agreement_stats):
    """
    Create a publication-quality heatmap showing agreement percentages.
    """
    # Pivot the data for the heatmap
    heatmap_data = agreement_stats.pivot(
        index='User',
        columns='Benchmark Annotation',
        values='Agreement Percentage'
    ).fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set publication-friendly style
    sns.set_style("white")

    # Create the heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={'label': 'Agreement (%)', 'shrink': 0.8},
        ax=ax
    )

    # Customize the plot
    ax.set_title('Agreement Percentage Heatmap', fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    return fig


def process_dataset(file_path, dataset_name, cube_filter=None):
    """Process a single dataset and return the agreement stats"""
    try:
        # Read the file directly as text
        with open(file_path, 'r') as f:
            content = f.read()

        # Process the data
        data = []

        # Split content by lines and skip header
        lines = content.strip().split('\n')[1:]

        # For each line in the file
        for i, line in enumerate(lines, 1):
            # Extract all quoted strings using a more robust approach
            quoted_strings = re.findall(r'"([^"]*)"', line)

            # Create a unique identifier for this position
            position_id = f"pos_{i}"

            # Process each quoted string
            for quoted_string in quoted_strings:
                # Check if this is a user annotation (contains ":")
                if ":" in quoted_string:
                    # Skip non-user annotations
                    if any(skip in quoted_string for skip in ["pushed false positives synapses", "cube3: pushed"]):
                        continue

                    # Extract user and annotation type
                    parts = quoted_string.split(":", 1)
                    if len(parts) == 2:
                        user, annotation_type = parts
                        user = user.strip()
                        annotation_type = annotation_type.strip()

                        # Extract cube number if available in the line
                        cube_match = re.search(r'cube(\d+)', line, re.IGNORECASE)
                        cube_number = f"cube{cube_match.group(1)}" if cube_match else "unknown"

                        # Add to data
                        data.append({
                            'post_x': i,  # Using line number as position
                            'post_y': 0,
                            'post_z': 0,
                            'position_id': position_id,
                            'user': user,
                            'other_annotations': quoted_string,
                            'execution_time': pd.Timestamp.now(),
                            'cube_number': cube_number
                        })

        # If no data was extracted, try a different approach
        if len(data) == 0:
            # Read the file line by line
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # For each line in the file (skip header)
            for i, line in enumerate(lines[1:], 1):
                # Split by tab or comma (depending on the file format)
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(',')

                # Clean up parts
                parts = [p.strip().strip('"') for p in parts if p.strip()]

                # Create a unique identifier for this position
                position_id = f"pos_{i}"

                # Process each part
                for part in parts:
                    # Check if this is a user annotation (contains ":")
                    if ":" in part:
                        # Skip non-user annotations
                        if any(skip in part for skip in ["pushed false positives synapses", "cube3: pushed"]):
                            continue

                        # Extract user and annotation type
                        user_parts = part.split(":", 1)
                        if len(user_parts) == 2:
                            user, annotation_type = user_parts
                            user = user.strip()
                            annotation_type = annotation_type.strip()

                            # Extract cube number if available in the line
                            cube_match = re.search(r'cube(\d+)', line, re.IGNORECASE)
                            cube_number = f"cube{cube_match.group(1)}" if cube_match else "unknown"

                            # Add to data
                            data.append({
                                'post_x': i,  # Using line number as position
                                'post_y': 0,
                                'post_z': 0,
                                'position_id': position_id,
                                'user': user,
                                'other_annotations': part,
                                'execution_time': pd.Timestamp.now(),
                                'cube_number': cube_number
                            })

        # If still no data, return None
        if len(data) == 0:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add dataset identifier column
        df['dataset'] = dataset_name

        # Apply cube filter if provided
        if cube_filter is not None:
            df = df[df['cube_number'] == cube_filter]
            if df.empty:
                return None

        # Get all users
        all_users = df['user'].unique()

        # Check if benchmark user 'ac' exists
        if 'ac' not in df['user'].unique():
            return None

        # Generate agreement stats
        agreement_stats = plot_user_agreement(df, all_users)

        return agreement_stats

    except Exception as e:
        st.error(f"Error processing {dataset_name}: {str(e)}")
        return None


def plot_user_overview(df, selected_users):
    # Get user counts and sort
    user_counts = df[df['user'].isin(selected_users)].groupby('user').size().sort_values(ascending=False)

    fig = go.Figure(data=[
        go.Bar(
            x=user_counts.index,
            y=user_counts.values,
            marker_color='#00B4D8',
            width=0.8
        )
    ])

    fig.update_layout(
        template='plotly_dark',
        title=f'Review contributions by users on {selected_dataset}',
        title_x=0.001,
        title_font_size=20,
        xaxis_title='User',
        yaxis_title='Number of transactions',
        xaxis_tickangle=45,
        showlegend=False,
        height=500,
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.2)',
            gridwidth=1,
            griddash='dash'
        )
    )

    return fig


def plot_classification_breakdown(df, selected_users):
    """
    Create a plot showing the percentage breakdown of synapse classifications by user.
    Classifications are mapped to standardized categories: correct, incorrect, uncertain/dubious.
    """
    # Create a copy of the dataframe to avoid modifying the original
    analysis_df = df[df['user'].isin(selected_users)].copy()

    # Debug: Print sample of other_annotations to understand format
    st.write("Sample annotations:", analysis_df['other_annotations'].head(5).tolist())

    # User initials mapping
    user_mapping = {
        'mc': 'mclayton',
        'acs': 'acorreia',
        'ad': 'adulac',
        'sh': 'sharris',
        'gmo': 'gmo',
        'ngc': 'nceffa',
        'sy': 'shiyan',
        'sw': 'swilson',
        'mr': 'mrobbins',
        'ac': 'ac'  # Benchmark user
    }

    # Define mapping function to standardize annotation types
    def map_annotation_type(row):
        annotation_text = str(row['other_annotations']).lower() if pd.notna(row['other_annotations']) else ""

        # Debug print for a few rows
        if np.random.random() < 0.01:  # Print ~1% of rows for debugging
            print(f"User: {row['user']}, Annotation: {annotation_text}")

        # Special handling for benchmark user 'ac'
        if row['user'] == 'acardona':
            # If 'dubious' is present anywhere, prioritize it over other classifications
            if 'dubious' in annotation_text:
                return 'Uncertain/Dubious'
            # Otherwise, check for other classifications
            elif 'correct' in annotation_text and 'incorrect' not in annotation_text:
                return 'Correct'
            elif 'incorrect' in annotation_text:
                return 'Incorrect'

        # Direct mapping for common patterns
        if 'pre correct' in annotation_text or (
                'correct' in annotation_text and 'incorrect' not in annotation_text):
            return 'Correct'
        elif 'wrong set' in annotation_text or 'incorrect' in annotation_text:
            return 'Incorrect'
        elif 'distanced set' in annotation_text or 'distance set' in annotation_text:
            return 'Incorrect'  # Counting distanced as incorrect
        elif 'uncertain' in annotation_text or 'dubious' in annotation_text:
            return 'Uncertain/Dubious'

        # If we get here, try to parse the format "user: annotation_type"
        if ':' in annotation_text:
            parts = annotation_text.split(':', 1)
            if len(parts) == 2:
                user_initial = parts[0].strip()
                annotation = parts[1].strip()

                # Special case for 'ac' user annotations
                if user_initial == 'acardona' and 'dubious' in annotation:
                    return 'Uncertain/Dubious'
                if 'pre correct' in annotation or (
                        'correct' in annotation and 'incorrect' not in annotation):
                    return 'Correct'
                elif 'wrong set' in annotation or 'incorrect' in annotation:
                    return 'Incorrect'
                elif 'distanced set' in annotation or 'distance set' in annotation:
                    return 'Incorrect'
                elif 'uncertain' in annotation or 'dubious' in annotation or 'unknown' in annotation:
                    return 'Uncertain/Dubious'
            elif len(parts) == 1:
                if 'uncertain' in annotation or 'dubious' in annotation or 'unknown' in annotation:
                    return 'Uncertain/Dubious'

        # Default case for unrecognized formats
        return 'Other'

    # Apply mapping to get standardized categories
    analysis_df['classification'] = analysis_df.apply(map_annotation_type, axis=1)

    # Debug: Show distribution of classifications
    st.write("Classification distribution:", analysis_df['classification'].value_counts())

    # Group by user and classification to get counts
    classification_counts = analysis_df.groupby(['user', 'classification']).size().reset_index(name='count')

    # Debug: Print classification counts
    st.write("Classification counts by user:", classification_counts)

    # Calculate total annotations per user
    user_totals = classification_counts.groupby('user')['count'].sum().reset_index(name='total')

    # Merge to get percentages
    classification_pct = pd.merge(classification_counts, user_totals, on='user')
    classification_pct['percentage'] = (classification_pct['count'] / classification_pct['total'] * 100).round(1)

    # Calculate average percentage for 'Correct' classifications
    correct_avg = classification_pct[classification_pct['classification'] == 'Correct']['percentage'].mean()

    # Create the plot
    fig = px.bar(
        classification_pct,
        x='user',
        y='percentage',
        color='classification',
        color_discrete_map={
            'Correct': '#4CAF50',  # Green
            'Incorrect': '#F44336',  # Red
            'Uncertain/Dubious': '#FFC107',  # Amber
            'Other': '#9E9E9E'  # Gray
        },
        title='Synapse Classification Breakdown by User',
        labels={
            'user': 'User',
            'percentage': 'Percentage (%)',
            'classification': 'Classification'
        },
        template='plotly_dark',
        barmode='stack'
    )
    # Add average line for 'Correct' classifications
    fig.add_shape(
        type="line",
        x0=-0.5,  # Start before first bar
        y0=correct_avg,
        x1=len(classification_pct['user'].unique()) - 0.5,  # End after last bar
        y1=correct_avg,
        line=dict(
            color="#FFFFFF",
            width=2,
            dash="dash",
        ),
    )

    # Add annotation for the average line
    fig.add_annotation(
        x=len(classification_pct['user'].unique()) - 0.5,  # Position at the end
        y=correct_avg,
        text=f"Avg Correct: {correct_avg:.1f}%",
        showarrow=False,
        font=dict(
            size=12,
            color="#FFFFFF"
        ),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="#FFFFFF",
        borderwidth=1,
        borderpad=4,
        xshift=10
    )

    # Customize layout
    fig.update_layout(
        xaxis_tickangle=45,
        legend_title_text='Classification',
        height=500,
        yaxis=dict(
            title='Percentage (%)',
            range=[0, 100],
            gridcolor='rgba(255, 255, 255, 0.2)',
            gridwidth=1,
            griddash='dash'
        )
    )

    return fig

def plot_cube_details(df, selected_cube_number, selected_users):
    # cube_data = df[
    #     (df['cube'] == selected_cube) &
    #     (df['user'].isin(selected_users))
    # ]
    # Match the full cube identifier using the number
    cube_data = df[
        (df['cube_number'] == selected_cube_number) &
        (df['user'].isin(selected_users))
        ]

    # User agreement analysis
    users_in_cube = cube_data['user'].unique()
    total_annotations = len(cube_data)

    # Create temporal plot
    fig_temporal = go.Figure()

    for user in users_in_cube:
        user_data = cube_data[cube_data['user'] == user]
        fig_temporal.add_trace(go.Scatter(
            x=user_data['execution_time'],
            y=[user] * len(user_data),
            name=user,
            mode='markers+lines',
            marker=dict(size=10)
        ))

    fig_temporal.update_layout(
        template='plotly_dark',
        title=f'Temporal Distribution of Annotations in Cube {selected_cube_number}',
        xaxis_title='Time',
        yaxis_title='Users',
        height=400
    )

    return fig_temporal, total_annotations, users_in_cube


def plot_temporal_analysis(df, selected_users):
    # Create figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot transactions for each user
    for user in selected_users:
        user_data = df[df['user'] == user].sort_values('execution_time')
        line = plt.plot(user_data['execution_time'],
                        range(len(user_data)),
                        'o-',
                        label=user,
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)

        # Add count annotation near the last point
        if len(user_data) > 0:
            last_x = user_data['execution_time'].iloc[-1]
            last_y = len(user_data) - 1
            plt.annotate(f'n={len(user_data)}',
                         xy=(last_x, last_y),
                         xytext=(10, 0),
                         textcoords='offset points',
                         fontsize=12,
                         fontweight='bold',
                         color=line[0].get_color())

    # Customize appearance
    plt.xlabel('Time', fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel('Cumulative Transactions', fontsize=16, fontweight='bold', labelpad=15)

    # Format title with time range and total count
    start_time = df['execution_time'].min().strftime('%H:%M')
    end_time = df['execution_time'].max().strftime('%H:%M')
    total_count = len(df)
    plt.title(f'Transactions by User (Total: {total_count})\n{start_time} - {end_time}',
              fontsize=20, fontweight='bold', pad=20)

    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add break time region
    break_start = pd.to_datetime(df['execution_time'].iloc[0].strftime('%Y-%m-%d') + ' 18:00:00')
    break_end = pd.to_datetime(df['execution_time'].iloc[0].strftime('%Y-%m-%d') + ' 18:30:00')
    plt.axvspan(break_start, break_end, color='red', alpha=0.2, label='Break time')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1),
               loc='upper left',
               fontsize=12,
               frameon=False)

    plt.tight_layout()

    return fig


# In the main Streamlit app
# Add Leaderboard tab content
with tab1:
    st.header("🏆 Annotation Leaderboard")

    # Calculate user stats for leaderboard
    leaderboard_data = df[df['user'].isin(selected_users)].copy()

    # Group by user and cube to get stats
    user_cube_stats = leaderboard_data.groupby(['user', 'cube_number']).agg({
        'execution_time': ['count', 'min', 'max']
    }).reset_index()

    # Flatten the column hierarchy
    user_cube_stats.columns = ['User', 'Cube', 'Annotations', 'First Annotation', 'Last Annotation']

    # Calculate time spent (in minutes)
    user_cube_stats['Time Spent (min)'] = (user_cube_stats['Last Annotation'] -
                                           user_cube_stats['First Annotation']).dt.total_seconds() / 60
    user_cube_stats['Time Spent (min)'] = user_cube_stats['Time Spent (min)'].round(2)

    # Calculate annotation rate (annotations per minute)
    user_cube_stats['Annotations/min'] = (user_cube_stats['Annotations'] /
                                          user_cube_stats['Time Spent (min)']).round(2)

    # Replace NaN values (when time spent is 0) with annotation count
    user_cube_stats['Annotations/min'] = user_cube_stats['Annotations/min'].fillna(
        user_cube_stats['Annotations'])

    # Sort by annotations count (descending)
    leaderboard = user_cube_stats.sort_values('Annotations', ascending=False)

    # Display top performers
    st.subheader("Top Performers by Annotation Count")
    st.dataframe(
        leaderboard[['User', 'Cube', 'Annotations', 'Time Spent (min)', 'Annotations/min']],
        use_container_width=True,
        hide_index=True
    )

    # Visualization of top performers
    top_users = leaderboard.head(10)

    fig = px.bar(
        top_users,
        x='User',
        y='Annotations',
        color='Cube',
        hover_data=['Time Spent (min)', 'Annotations/min'],
        title='Top 10 Performers by Annotation Count',
        template='plotly_dark'
    )

    fig.update_layout(
        xaxis_title='User',
        yaxis_title='Number of Annotations',
        xaxis_tickangle=45,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Efficiency leaderboard (annotations per minute)
    st.subheader("Most Efficient Annotators")

    # Filter out entries with very short time spans that might skew efficiency metrics
    efficiency_leaderboard = leaderboard[leaderboard['Time Spent (min)'] >= 1].sort_values(
        'Annotations/min', ascending=False)

    if not efficiency_leaderboard.empty:
        fig_efficiency = px.bar(
            efficiency_leaderboard.head(10),
            x='User',
            y='Annotations/min',
            color='Cube',
            hover_data=['Annotations', 'Time Spent (min)'],
            title='Top 10 Most Efficient Annotators',
            template='plotly_dark'
        )

        fig_efficiency.update_layout(
            xaxis_title='User',
            yaxis_title='Annotations per Minute',
            xaxis_tickangle=45,
            height=500
        )

        st.plotly_chart(fig_efficiency, use_container_width=True)
    else:
        st.info("Not enough time data to calculate efficiency metrics.")

with tab2:
    st.header("Annotation Analysis")

    # First show overall user contributions
    user_overview = plot_user_overview(df, selected_users)
    st.plotly_chart(user_overview, use_container_width=True)

    # Then show Dataset Breakdown
    st.subheader(" Dataset --> Cube-wise Num of Annotations")

    # Create simplified cube numbers
    # print(df['cube'].unique())
    # df['cube_number'] = df['cube'].apply(lambda x: f"cube#{x[-1]}")

    # Get total annotations per cube
    cube_counts = df.groupby('cube_number').size().sort_values(ascending=False)

    # Plot cube-wise distribution
    fig_cube = go.Figure(data=[
        go.Bar(
            x=cube_counts.index,
            y=cube_counts.values,
            marker_color='#00B4D8',
            width=0.8
        )
    ])

    fig_cube.update_layout(
        template='plotly_dark',
        title=f'Annotations per cube in {selected_dataset}',
        title_x=0.001,
        title_font_size=20,
        xaxis_title='Cube Number',
        yaxis_title='Number of Annotations',
        xaxis_tickangle=45,
        showlegend=False,
        height=400,
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.2)',
            gridwidth=1,
            griddash='dash'
        )
    )

    st.plotly_chart(fig_cube, use_container_width=True)

    # Allow cube selection for detailed analysis
    selected_cube = st.selectbox("Select cube for detailed analysis", cube_counts.index)
    if selected_cube:
        col1, col2, col3 = st.columns(3)
        temporal_fig, total_annot, users = plot_cube_details(df, selected_cube, selected_users)

        # Display metrics
        with col1:
            st.metric("Total Annotations", total_annot)
        with col2:
            st.metric("Number of Users", len(users))
        with col3:
            # Update to use cube_number for filtering
            all_positions = df[df['cube_number'] == selected_cube].groupby(['post_x', 'post_y', 'post_z']).agg({
                'user': 'nunique'
            }).reset_index()
            if len(all_positions) > 0:  # Add check for empty dataframe
                agreement_points = len(all_positions[all_positions['user'] > 1])
                agreement_pct = round((agreement_points / len(all_positions)) * 100, 1)
                st.metric("Agreement Points (%)", f"{agreement_pct}%")
            else:
                st.metric("Agreement Points (%)", "0%")

        # Show temporal distribution
        st.plotly_chart(temporal_fig, use_container_width=True)

        # Show agreement details
        st.subheader("Position Agreement Details")
        agreement_data = df[df['cube_number'] == selected_cube].groupby(['post_x', 'post_y', 'post_z']).agg({
            'user': lambda x: list(x),
            'other_annotations': lambda x: list(x),
            'execution_time': 'count'
        }).reset_index()

        agreement_data['users'] = agreement_data['user'].apply(lambda x: ', '.join(x))
        agreement_data['num_users'] = agreement_data['user'].apply(len)

        # Filter and display agreement data
        agreement_view = agreement_data[['other_annotations', 'post_x', 'post_y', 'post_z', 'num_users', 'users']]
        agreement_view.columns = ['annotations', 'post_x', 'post_y', 'post_z', 'Number of Users', 'Users']
        st.dataframe(agreement_view.sort_values('Number of Users', ascending=False))

with tab3:
    # st.header("User Agreement Analysis")

    # # Calculate position agreement
    # def get_position_agreement(df):
    #     position_agreement = defaultdict(list)

    #     for _, group in df.groupby(['post_x', 'post_y', 'post_z']):
    #         users = group['user'].unique()
    #         if len(users) > 1:  # Only consider positions marked by multiple users
    #             position_agreement['position'].append(f"({group['post_x'].iloc[0]}, {group['post_y'].iloc[0]}, {group['post_z'].iloc[0]})")
    #             position_agreement['num_users'].append(len(users))
    #             position_agreement['users'].append(', '.join(users))

    #     return pd.DataFrame(position_agreement)

    # agreement_df = get_position_agreement(df[df['user'].isin(selected_users)])

    # # Plot agreement distribution
    # fig = px.histogram(agreement_df,
    #                   x='num_users',
    #                   title='Distribution of User Agreement on Positions',
    #                   labels={'num_users': 'Number of Users Agreeing', 'count': 'Frequency'})

    # st.plotly_chart(fig, use_container_width=True)

    # # Show detailed agreement table
    # st.dataframe(agreement_df)

    st.header("User Agreement Analysis")

    st.info("**Note:** User 'ac' is our benchmark baseline for annotation agreement.")

    # Add explanation of the agreement analysis
    st.markdown("""
    ### How to Interpret the Agreement Analysis

    - The heatmap shows how often each user's annotations agree with the benchmark (user 'ac').
    - Higher percentages (green) indicate stronger agreement with the benchmark.
    - Lower percentages (red) indicate areas where users differ from the benchmark.
    - The analysis is broken down by annotation type (correct, incorrect, dubious).
    """)

    # Define data directory for benchmark files
    data_dir = './data'

    # Create file paths based on selected dataset
    if selected_dataset == "Octo False Positives":
        benchmark_file = os.path.join(data_dir, "benchmark_against_albert_octo_cube3.csv")
        dataset_name = "Octo False Positives"
    elif selected_dataset == "MR143 False Positives":
        benchmark_file = os.path.join(data_dir, "final_df_postsyn_transaction_mr143.csv")
        dataset_name = "MR143 False Positives"
    else:
        benchmark_file = None
        dataset_name = None

    # Process benchmark file if it exists
    if benchmark_file and os.path.exists(benchmark_file):
        st.subheader(f"Agreement Analysis for {dataset_name}")

        # Add option to filter by cube if needed
        if 'cube_number' in df.columns and len(df['cube_number'].unique()) > 1:
            use_cube_filter = st.checkbox("Filter by cube", value=False)
            if use_cube_filter:
                cube_filter = selected_cube if 'selected_cube' in locals() else st.selectbox(
                    "Select cube for agreement analysis",
                    sorted(df['cube_number'].unique())
                )
                st.info(f"Showing agreement for cube: {cube_filter}")
                # We'll pass this to process_dataset later
            else:
                cube_filter = None
                st.info("Showing agreement across Octo cube#3")
        else:
            use_cube_filter = False
            cube_filter = None

        # Process the dataset and get agreement stats
        with st.spinner("Calculating user agreement..."):
            agreement_stats = process_dataset(benchmark_file, dataset_name, cube_filter)

            if agreement_stats is not None and not agreement_stats.empty:
                # Display the heatmap
                fig = plot_agreement_heatmap(agreement_stats)
                st.pyplot(fig)

                # Display the raw data with improved formatting
                st.subheader("Agreement Statistics")

                # Format the dataframe for better display
                display_stats = agreement_stats.copy()
                display_stats['Agreement Percentage'] = display_stats['Agreement Percentage'].apply(
                    lambda x: f"{x:.1f}%")

                st.dataframe(
                    display_stats,
                    column_config={
                        "User": st.column_config.TextColumn("User"),
                        "Benchmark Annotation": st.column_config.TextColumn("Benchmark Type"),
                        "Total Annotations": st.column_config.NumberColumn("Total"),
                        "Agreements": st.column_config.NumberColumn("Matches"),
                        "Agreement Percentage": st.column_config.TextColumn("Agreement")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(
                    "No agreement data could be calculated. This might be because there are no benchmark annotations from user 'ac' or no matching positions.")
    else:
        st.warning(f"Benchmark file for {selected_dataset} not found.")
        st.info("Please select a dataset with benchmark annotations to view agreement analysis.")

with tab4:
    st.header("Temporal Analysis")

    # Convert execution_time to datetime if not already
    df['execution_time'] = pd.to_datetime(df['execution_time'])

    # Get min and max dates from the data
    min_date = df['execution_time'].min().date()
    max_date = df['execution_time'].max().date()

    # Time range selector with better error handling
    try:
        # date_range = st.date_input(
        #     "Select Date Range",
        #     value=(min_date, max_date),
        #     min_value=min_date,
        #     max_value=max_date
        # )

        # Use a simpler approach with two separate date inputs
        st.subheader("Select Date Range")
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)

        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        # Filter by date range
        mask = (df['execution_time'].dt.date >= start_date) & (df['execution_time'].dt.date <= end_date)
        time_filtered_df = df[mask & df['user'].isin(selected_users)]

        if time_filtered_df.empty:
            st.warning("No data available for the selected date range.")
            time_filtered_df = df[df['user'].isin(selected_users)]  # Use all data as fallback

    except Exception as e:
        st.error(f"Error with date selection: {e}")
        # Default to using all data if there's an error
        time_filtered_df = df[df['user'].isin(selected_users)]

    # Create and display the temporal plot
    if not time_filtered_df.empty:
        fig = plot_temporal_analysis(time_filtered_df, selected_users)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected users in this time range.")

    # Add interactive time range selection if needed
    start_hour = st.slider("Start Hour", 0, 23, 16)
    end_hour = st.slider("End Hour", 0, 23, 19)

    # Filter data based on hour selection
    filtered_df = df[
        (df['execution_time'].dt.hour >= start_hour) &
        (df['execution_time'].dt.hour <= end_hour)
        ]

    if not filtered_df.empty:
        fig_filtered = plot_temporal_analysis(filtered_df, selected_users)
        st.pyplot(fig_filtered)
    else:
        st.warning("No data available for the selected time range.")

with tab5:
    st.header("Summary Statistics")

    # Calculate more accurate metrics
    filtered_df = df[df['user'].isin(selected_users)]

    # Basic metrics in cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Annotations", len(filtered_df))

    with col2:
        st.metric("Active Users", len(selected_users))

    with col3:
        median_annotations = int(filtered_df.groupby('user').size().median())
        st.metric("Median Annotations per User", median_annotations)

    # Add synapse classification breakdown
    st.subheader("Synapse Classification Breakdown")
    st.markdown("""
    This chart shows how each user classifies synapses as a percentage of their total annotations:
    - **Correct**: Includes 'pre correct', 'correct'
    - **Incorrect**: Includes 'wrong set', 'incorrect', 'distanced set'
    - **Uncertain/Dubious**: Includes 'uncertain', 'dubious'
    """)

    classification_fig = plot_classification_breakdown(df, selected_users)
    st.plotly_chart(classification_fig, use_container_width=True)

    # Add more detailed statistics
    st.subheader("User Activity Breakdown")

    # Create a more comprehensive user stats dataframe
    user_stats = filtered_df.groupby('user').agg({
        'execution_time': ['count', 'min', 'max'],
        'cube_number': 'nunique'
    }).reset_index()

    # Rename columns for clarity
    user_stats.columns = ['User', 'Annotation Count', 'First Activity', 'Last Activity', 'Cubes Worked On']

    # Calculate active time (time between first and last annotation)
    user_stats['Active Period (hours)'] = (user_stats['Last Activity'] - user_stats[
        'First Activity']).dt.total_seconds() / 3600
    user_stats['Active Period (hours)'] = user_stats['Active Period (hours)'].round(2)

    # Calculate hourly rate (more meaningful than daily for short sessions)
    user_stats['Annotations per Hour'] = user_stats['Annotation Count'] / user_stats['Active Period (hours)']
    user_stats['Annotations per Hour'] = user_stats['Annotations per Hour'].round(2)

    # Handle cases where active period is very short (< 0.1 hour)
    user_stats.loc[user_stats['Active Period (hours)'] < 0.1, 'Annotations per Hour'] = user_stats['Annotation Count']

    # Display the detailed stats table
    st.dataframe(
        user_stats[['User', 'Annotation Count', 'Cubes Worked On', 'Active Period (hours)', 'Annotations per Hour']],
        use_container_width=True,
        hide_index=True
    )

    # Create more meaningful visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Annotation count by user
        fig1 = px.bar(
            user_stats.sort_values('Annotation Count', ascending=False),
            x='User',
            y='Annotation Count',
            title='Total Annotations by User',
            template='plotly_dark',
            color='Annotation Count',
            color_continuous_scale='Viridis'
        )

        fig1.update_layout(
            xaxis_title='User',
            yaxis_title='Number of Annotations',
            xaxis_tickangle=45,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Annotations per hour (better productivity metric)
        fig2 = px.bar(
            user_stats.sort_values('Annotations per Hour', ascending=False),
            x='User',
            y='Annotations per Hour',
            title='Annotation Speed (per hour)',
            template='plotly_dark',
            color='Annotations per Hour',
            color_continuous_scale='Viridis'
        )

        fig2.update_layout(
            xaxis_title='User',
            yaxis_title='Annotations per Hour',
            xaxis_tickangle=45,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Add a pie chart showing distribution of annotations across users
    user_distribution = filtered_df.groupby('user').size().reset_index()
    user_distribution.columns = ['User', 'Count']

    fig3 = px.pie(
        user_distribution,
        values='Count',
        names='User',
        title='Distribution of Annotations by User',
        template='plotly_dark',
        hole=0.4
    )

    fig3.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig3, use_container_width=True)

# Add a new tab for media examples
with tab6:
    st.header("📷 Synapse Media Examples")

    # Create categories for different types of examples
    example_type = st.radio(
        "Select Example Type",
        ["Good Synapses", "False Positives", "Ambiguous Cases"],
        horizontal=True
    )

    # Create columns for better layout
    col1, col2 = st.columns(2)


    # Function to display CATMAID URLs with context
    def display_catmaid_example(title, image_path=None, catmaid_url=None, description=None):
        st.subheader(title)

        if image_path:
            st.image(image_path, caption=title, use_container_width=True)

        if description:
            st.markdown(f"**Description:** {description}")

        if catmaid_url:
            st.markdown(f"[View in CATMAID]({catmaid_url}) (requires access)")
            st.info(
                "Note: Viewing in CATMAID requires appropriate access permissions. Synapses/Not are centered in the images.")


    # Display examples based on selected type
    if example_type == "Good Synapses":
        with col1:
            display_catmaid_example(
                "Example 1:",
                image_path="./data/examples_images/good_synapse_neuron394147.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/ym86axf",
                # description="Classic T-bar structure with multiple postsynaptic densities. Note the clear synaptic cleft and membrane specializations."
                description="Ground-Truth Synapse for Models."
            )

            display_catmaid_example(
                "Example 2:",
                image_path="./data/examples_images/good_synapse_neuron416101.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/bla62bn",
                # description="Synapse with multiple postsynaptic partners clearly visible."
                description="Ground-Truth Synapse for Models."

            )

            display_catmaid_example(
                "Example 3:",
                image_path="./data/examples_images/good_synapse_neuron416117.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/w6ccxwv",
                # description="Synapse with multiple postsynaptic partners clearly visible."
                description="Ground-Truth Synapse for Models."

            )

        # with col2:
        #     display_catmaid_example(
        #         "Example 2: Pedunculated T-bar",
        #         image_path="./data/example_images/good_synapse_2.png",
        #         catmaid_url="https://catmaid-example-url.org/2",
        #         description="T-bar with a clear peduncle and platform. Note the vesicle clustering."
        #     )

        #     display_catmaid_example(
        #         "Example 4: Small but Clear Synapse",
        #         image_path="./data/example_images/good_synapse_4.png",
        #         catmaid_url="https://catmaid-example-url.org/4",
        #         description="Smaller synapse but with all defining characteristics present."
        #     )

    elif example_type == "False Positives":
        with col1:
            display_catmaid_example(
                "Example 1:",
                image_path="./data/examples_images/bad_synapse_neuron416164.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/tchji7f",
                description="Marked by Albert as False Positives in Octo"
                # description="Dense core vesicle that might be mistaken for a small T-bar. Note the lack of synaptic cleft and postsynaptic density."
            )

            display_catmaid_example(
                "Example 2:",
                image_path="./data/examples_images/bad_synapse_neuron416535.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/9jttfqh",
                description="Marked by Albert as False Positives in Octo"
            )

            display_catmaid_example(
                "Example 3:",
                image_path="./data/examples_images/bad_synapse_neuron416806.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/ygfz9fb",
                description="Marked by Albert as False Positives in Octo"
            )

        # with col2:
        #     display_catmaid_example(
        #         "Example 2: Membrane Artifact",
        #         image_path="./data/example_images/false_positive_2.png",
        #         catmaid_url="https://catmaid-example-url.org/fp2",
        #         description="Membrane artifact with density that resembles a synapse. Note the lack of vesicle clustering."
        #     )

        #     display_catmaid_example(
        #         "Example 4: Section Boundary Artifact",
        #         image_path="./data/example_images/false_positive_4.png",
        #         catmaid_url="https://catmaid-example-url.org/fp4",
        #         description="Section boundary artifact that can be mistaken for synaptic density."
        #     )

    else:  # Ambiguous Cases
        with col1:
            display_catmaid_example(
                "Example 1:",
                image_path="./data/examples_images/ambiguous_neuron605629.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/18/links/52b72qi",
                description="Potential developing synapse with some but not all characteristics of a mature synapse."
            )

            display_catmaid_example(
                "Example 2: ",
                image_path="./data/examples_images/ambiguous_neuron606017.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/18/links/5ictpti",
                description="Oblique section through what might be a synapse, making classification difficult."
            )

        # with col2:
        #     display_catmaid_example(
        #         "Example 2: Poor Image Quality Region",
        #         image_path="./data/example_images/ambiguous_2.png",
        #         catmaid_url="https://catmaid-example-url.org/amb2",
        #         description="Region with poor image quality where synaptic structures are difficult to identify with certainty."
        #     )

        #     display_catmaid_example(
        #         "Example 4: Unusual Morphology",
        #         image_path="./data/example_images/ambiguous_4.png",
        #         catmaid_url="https://catmaid-example-url.org/amb4",
        #         description="Structure with unusual morphology that has some synaptic characteristics but doesn't fit typical patterns."
        #     )

    # Add option to upload custom examples
    st.divider()
    st.subheader("Add Your Own Examples")

    uploaded_file = st.file_uploader("Upload an image example", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Example", use_column_width=True)

        # Option to save the uploaded image
        example_title = st.text_input("Example Title")
        example_description = st.text_area("Example Description")
        example_category = st.selectbox("Category", ["Good Synapse", "False Positive", "Ambiguous"])

        if st.button("Save Example"):
            # Code to save the example would go here
            # This would require file system access to save the image
            st.success("Example saved successfully!")

# Replace with your actual app URL
app_url = "https://synapse-proofreading-dashboard.streamlit.app"
qr_img = generate_qr_code(app_url)

st.sidebar.markdown(f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{qr_img}' width='180'>
        <p style='font-size: 12px;'>Scan to open on mobile</p>
    </div>
    <div style='text-align: center; margin-top: 20px; font-size: 12px;'>
        © 2023 University of Cambridge<br>
        All rights reserved
    </div>
""", unsafe_allow_html=True)

# Add download button for filtered data
st.sidebar.download_button(
    label="Download Filtered Data",
    data=df[df['user'].isin(selected_users)].to_csv(index=False).encode('utf-8'),
    file_name="filtered_synapse_data.csv",
    mime="text/csv"
)

if __name__ == '__main__':
    pass