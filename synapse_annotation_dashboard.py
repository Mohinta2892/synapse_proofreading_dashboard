import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter 


# Set page config
st.set_page_config(
    page_title="Synapse Annotation Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load data
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Octo False Positives":
        df = pd.read_csv('./data/final_df_postsyn_transaction_octo.csv')
    elif dataset_name == "MR143 False Positives":  # Add more datasets as needed
        df = pd.read_csv('./data/final_df_postsyn_transaction_mr143.csv')
    
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
df['cube_ds_pos'] = df.apply(lambda x: f"{x['dataset']}_({int(x['post_x']//100)}, {int(x['post_y']//100)}, {int(x['post_z']//100)})", axis=1)

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
st.title("🧠 Synapse Annotation Analysis Dashboard")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Leaderboard", "Cube Analysis", "User Agreement", "Time Analysis", "Statistics", "Media Examples"])
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
        title_x=0.5,
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
        title_x=0.5,
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
        agreement_view = agreement_data[['other_annotations','post_x', 'post_y', 'post_z', 'num_users', 'users']]
        agreement_view.columns = ['annotations', 'post_x', 'post_y', 'post_z', 'Number of Users', 'Users']
        st.dataframe(agreement_view.sort_values('Number of Users', ascending=False))

with tab3:
    st.header("User Agreement Analysis")
    
    # Calculate position agreement
    def get_position_agreement(df):
        position_agreement = defaultdict(list)
        
        for _, group in df.groupby(['post_x', 'post_y', 'post_z']):
            users = group['user'].unique()
            if len(users) > 1:  # Only consider positions marked by multiple users
                position_agreement['position'].append(f"({group['post_x'].iloc[0]}, {group['post_y'].iloc[0]}, {group['post_z'].iloc[0]})")
                position_agreement['num_users'].append(len(users))
                position_agreement['users'].append(', '.join(users))
        
        return pd.DataFrame(position_agreement)
    
    agreement_df = get_position_agreement(df[df['user'].isin(selected_users)])
    
    # Plot agreement distribution
    fig = px.histogram(agreement_df, 
                      x='num_users',
                      title='Distribution of User Agreement on Positions',
                      labels={'num_users': 'Number of Users Agreeing', 'count': 'Frequency'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed agreement table
    st.dataframe(agreement_df)

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
    user_stats['Active Period (hours)'] = (user_stats['Last Activity'] - user_stats['First Activity']).dt.total_seconds() / 3600
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
    col1, col2 = st.columns(1)
    
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
            st.info("Note: Viewing in CATMAID requires appropriate access permissions. Synapses/Not are centered in the images.")
    
    # Display examples based on selected type
    if example_type == "Good Synapses":
        with col1:
            display_catmaid_example(
                "Example 1: ",
                image_path="./data/examples_images/good_synapse_neuron394147.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/ym86axf",
                # description="Classic T-bar structure with multiple postsynaptic densities. Note the clear synaptic cleft and membrane specializations."
                description="Ground-Truth Synapse for Models."
            )
            
            display_catmaid_example(
                "Example 2: ",
                image_path="./data/examples_images/good_synapse_neuron416101.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/bla62bn",
                # description="Synapse with multiple postsynaptic partners clearly visible."
                description="Ground-Truth Synapse for Models."

            )

            display_catmaid_example(
                "Example 3: ",
                image_path="./data/examples_images/good_synapse_neuron416117.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/9/links/w6ccxwv",
                # description="Synapse with multiple postsynaptic partners clearly visible."
                description="Ground-Truth Synapse for Models."

            )
            
            
        # with col2:
        #     display_catmaid_example(
        #         "Example 2: Pedunculated T-bar",
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/good_synapse_2.png",
        #         catmaid_url="https://catmaid-example-url.org/2",
        #         description="T-bar with a clear peduncle and platform. Note the vesicle clustering."
        #     )
            
        #     display_catmaid_example(
        #         "Example 4: Small but Clear Synapse",
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/good_synapse_4.png",
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
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/false_positive_2.png",
        #         catmaid_url="https://catmaid-example-url.org/fp2",
        #         description="Membrane artifact with density that resembles a synapse. Note the lack of vesicle clustering."
        #     )
            
        #     display_catmaid_example(
        #         "Example 4: Section Boundary Artifact",
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/false_positive_4.png",
        #         catmaid_url="https://catmaid-example-url.org/fp4",
        #         description="Section boundary artifact that can be mistaken for synaptic density."
        #     )
            
    else:  # Ambiguous Cases
        with col1:
            display_catmaid_example(
                "Example 1: ",
                image_path="./data/examples_images/ambiguous_neuron605629.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/18/links/52b72qi",
                description="Potential developing synapse with some but not all characteristics of a mature synapse."
            )
            
            display_catmaid_example(
                "Example 2: ",
                image_path="./data/examples_images/ambiguous_neuron606017.png",
                catmaid_url="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/18/links/5ictpti",
                description="Marked by Albert as an uncertain case."
            )
            
        # with col2:
        #     display_catmaid_example(
        #         "Example 2: Poor Image Quality Region",
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/ambiguous_2.png",
        #         catmaid_url="https://catmaid-example-url.org/amb2",
        #         description="Region with poor image quality where synaptic structures are difficult to identify with certainty."
        #     )
            
        #     display_catmaid_example(
        #         "Example 4: Unusual Morphology",
        #         image_path="/Users/sam/Library/CloudStorage/OneDrive-UniversityofCambridge/Synapse_localisation/synapse_curation/example_images/ambiguous_4.png",
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

# Add download button for filtered data
st.sidebar.download_button(
    label="Download Filtered Data",
    data=df[df['user'].isin(selected_users)].to_csv(index=False).encode('utf-8'),
    file_name="filtered_synapse_data.csv",
    mime="text/csv"
)

if __name__ == '__main__':
    pass