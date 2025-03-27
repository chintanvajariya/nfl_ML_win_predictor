import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Page and Title Setup
st.set_page_config(layout="wide")
st.title("NFL Win Probability Dashboard")
st.caption("Explore actual vs. predicted win probabilities with interactive model settings and feature selection.")

# Sidebar: Select Which Seasons to Include
st.sidebar.header("Data Years")

temp_selected_year = st.sidebar.selectbox(
    "Select Season:",
    options=[2023, 2022, 2021, 2020, 2019, 2018],
    index=0
)

# Loading Data (Cached)
@st.cache_data(show_spinner=True)
def load_data(years):
    df = nfl.import_pbp_data(
        years=years,
        columns=[
            'game_id', 'posteam', 'defteam', 'home_team', 'away_team',
            'posteam_score', 'defteam_score', 'yardline_100', 'game_date',
            'quarter_seconds_remaining', 'game_seconds_remaining',
            'down', 'wp', 'score_differential', 'ydstogo', 'week', 'stadium'
        ]
    )
    return df

effective_years = st.session_state.get("selected_year", [2023])  # default fallback
df = load_data(effective_years)

# Score Validation Function
def validate_scores(df):
    valid_point_increments = {1, 2, 3, 6, 7, 8}
    df['home_team_score'] = df['posteam_score'].where(df['posteam'] == df['home_team'], df['defteam_score'])
    df['away_team_score'] = df['defteam_score'].where(df['posteam'] == df['home_team'], df['posteam_score'])

    home_incr = df['home_team_score'].diff().fillna(0)
    away_incr = df['away_team_score'].diff().fillna(0)
    invalid = (
        (home_incr < 0) |
        (away_incr < 0) |
        (~home_incr.isin(valid_point_increments.union({0}))) |
        (~away_incr.isin(valid_point_increments.union({0})))
    )
    return df[~invalid].reset_index(drop=True)

# Preprocessing & Feature Engineering
valid_df = df[df['wp'].notnull() & df['posteam_score'].notnull() & df['defteam_score'].notnull()].copy()
valid_df['field_position'] = valid_df['yardline_100']
valid_df['possession_status'] = (valid_df['posteam'] == valid_df['home_team']).astype(int)
valid_df.fillna(0, inplace=True)
valid_df = validate_scores(valid_df)

# Normalize WP to home team’s perspective
valid_df['wp_normalized'] = np.where(
    valid_df['posteam'] == valid_df['home_team'],
    valid_df['wp'],
    1 - valid_df['wp']
)

# Sidebar: Model Settings
st.sidebar.header("Model Settings")

n_estimators = st.sidebar.slider(
    "n_estimators (Number of NFL Experts) - More expert opinions can boost accuracy, but slow things down.",
    min_value=10, max_value=300, value=100, step=10
)

max_depth = st.sidebar.slider(
    "max_depth (Expert Knowledge Depth) - Deeper knowledge helps spot complex patterns, but risks overthinking (overfitting).",
    min_value=4, max_value=30, value=16, step=2
)

# Sidebar: Feature Selection
st.sidebar.header("Feature Selection")

feature_mapping = {
    "Home Team Score": "home_team_score",
    "Away Team Score": "away_team_score",
    "Yards to Endzone": "field_position",
    "Possession Status": "possession_status",
    "Game Seconds Left": "game_seconds_remaining",
    "Down": "down",
    "Quarter Seconds Left": "quarter_seconds_remaining",
    "Yards to Go": "ydstogo"
}

# Set default checked state for each feature
default_checked_features = {
    "Quarter Seconds Left": False,
    "Yards to Go": False
}

selected_features = []
for label, col in feature_mapping.items():
    default_checked = default_checked_features.get(label, True)  # Default to True unless specified
    if st.sidebar.checkbox(label, value=default_checked):
        selected_features.append(col)

if len(selected_features) == 0:
    selected_features = list(feature_mapping.values())

# Sidebar: Game Picker (Full Names, No Game ID)
st.sidebar.header("Game Picker")

# Load full team info
team_info = nfl.import_team_desc()[['team_abbr', 'team_name', 'team_color', 'team_color2', 'team_color3', 'team_color4', 'team_logo_espn']]
abbr_to_name = dict(zip(team_info['team_abbr'], team_info['team_name']))
name_to_abbr = dict(zip(team_info['team_name'], team_info['team_abbr']))

# Use full names in dropdowns
home_teams = sorted(valid_df['home_team'].map(abbr_to_name).unique())
selected_home_team_name = st.sidebar.selectbox("Select Home Team", home_teams, index=25)
selected_home_team = name_to_abbr[selected_home_team_name]

# Filter away team options based on selected home team
away_teams = valid_df[valid_df['home_team'] == selected_home_team]['away_team'].unique()
away_teams_full = sorted([abbr_to_name[abbr] for abbr in away_teams])
selected_away_team_name = st.sidebar.selectbox("Select Away Team", away_teams_full)
selected_away_team = name_to_abbr[selected_away_team_name]

# Filter dates and format them nicely
filtered_df = valid_df[
    (valid_df['home_team'] == selected_home_team) &
    (valid_df['away_team'] == selected_away_team)
]

# Format date nicely for the dropdown
filtered_df['game_date_str'] = pd.to_datetime(filtered_df['game_date']).dt.strftime('%A, %b %d, %Y')
date_map = dict(zip(filtered_df['game_date_str'], filtered_df['game_date']))
selected_game_date_str = st.sidebar.selectbox("Select Game Date", sorted(date_map.keys()))
selected_game_date = date_map[selected_game_date_str]

# Pull the game data
game_df = valid_df[
    (valid_df['home_team'] == selected_home_team) &
    (valid_df['away_team'] == selected_away_team) &
    (valid_df['game_date'] == selected_game_date)
].reset_index(drop=True)

# Normalize actual WP in game_df to home team’s perspective (for comparing with predictions)
game_df['wp_normalized'] = np.where(
    game_df['posteam'] == game_df['home_team'],
    game_df['wp'],
    1 - game_df['wp']
)

selected_game_id = game_df['game_id'].iloc[0]
game_week = game_df['week'].iloc[0]
home_team = selected_home_team
away_team = selected_away_team
game_date = selected_game_date
formatted_game_date = pd.to_datetime(game_date).strftime('%A, %b %d, %Y')


# Get logos
home_logo = team_info.loc[team_info['team_abbr'] == home_team, 'team_logo_espn'].values[0]
away_logo = team_info.loc[team_info['team_abbr'] == away_team, 'team_logo_espn'].values[0]

venue = game_df['stadium'].iloc[0]

def get_team_record(df, team_abbr, game_date):
    df = df[pd.to_datetime(df['game_date']) <= pd.to_datetime(game_date)]

    game_results = df.groupby('game_id').agg(
        home_team=('home_team', 'last'),
        away_team=('away_team', 'last'),
        home_score=('home_team_score', 'last'),
        away_score=('away_team_score', 'last')
    ).reset_index()

    team_games = game_results[
        (game_results['home_team'] == team_abbr) | (game_results['away_team'] == team_abbr)
    ]

    def did_team_win(row):
        if row['home_team'] == team_abbr:
            return row['home_score'] > row['away_score']
        elif row['away_team'] == team_abbr:
            return row['away_score'] > row['home_score']
        return False

    team_games['win'] = team_games.apply(did_team_win, axis=1)
    wins = team_games['win'].sum()
    losses = len(team_games) - wins

    return f"{wins}–{losses}"

home_record = get_team_record(valid_df, home_team, game_date)
away_record = get_team_record(valid_df, away_team, game_date)

home_team_color = team_info.loc[team_info['team_abbr'] == home_team, 'team_color'].values[0]
away_colors = team_info.loc[team_info['team_abbr'] == away_team, ['team_color', 'team_color2', 'team_color3', 'team_color4']].values[0]


def hex_to_color(hex_color, alpha=None, as_string=False):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    if alpha is not None:
        if as_string:
            return f'rgba({r}, {g}, {b}, {alpha})'
        else:
            return (r, g, b, alpha)
    return (r, g, b)

def colors_are_too_similar(color1, color2, threshold=60):
    r1, g1, b1  = hex_to_color(color1)
    r2, g2, b2 = hex_to_color(color2)
    distance = ((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)**0.5
    return distance < threshold

BAD_COLORS = ['#000000', '#ffffff']

for color in away_colors:
    if not colors_are_too_similar(color, home_team_color) and not any(
        colors_are_too_similar(color, bad, threshold=80) for bad in BAD_COLORS
    ):
        away_team_color = color
        break
else:
    away_team_color = away_colors[3]

home_rgba = hex_to_color(home_team_color, alpha=0.1, as_string=True)
away_rgba = hex_to_color(away_team_color, alpha=0.1, as_string=True)

# Time Format Toggle
time_format = st.sidebar.radio(
    "Select Time Display Format:",
    options=[ "Quarter and Time", "Seconds Remaining"],
    index=0  # Default to "Seconds Remaining"
)

def convert_to_quarter_time(seconds):
    quarter_length = 900
    if seconds is None or pd.isnull(seconds):
        return ""
    quarter = 4 - (seconds // quarter_length)
    if quarter < 1:
        quarter = 1
    quarter_seconds = seconds % quarter_length
    minutes = quarter_seconds // 60
    secs = quarter_seconds % 60
    return f"Q{int(quarter)} {int(minutes)}:{int(secs):02d}"

if time_format == "Quarter and Time":
    game_df['time_display'] = game_df['game_seconds_remaining'].apply(convert_to_quarter_time)
else:
    game_df['time_display'] = game_df['game_seconds_remaining'].astype(str)

# Sidebar Submit Button
st.sidebar.markdown("---")
run_button_clicked = st.sidebar.button("Submit Changes")
print(nfl.import_team_desc().columns.to_list())

# Use session state to detect first run
if "first_run_done" not in st.session_state:
    st.session_state.first_run_done = True
    run_model = True  # Initial run
else:
    run_model = run_button_clicked  # Only run after user presses button

# Model Training (Cached)
@st.cache_data(show_spinner=True)
def train_model(features, n_est, m_depth):
    X = valid_df[features].dropna()
    y = valid_df['wp_normalized'].loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_ = RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42)
    model_.fit(X_train, y_train)
    mse_ = mean_squared_error(y_test, model_.predict(X_test))
    return model_, mse_

if run_model:
    model, mse = train_model(selected_features, n_estimators, max_depth)

    imputer = SimpleImputer(strategy='mean')
    game_X = game_df[selected_features]
    game_X_imputed = pd.DataFrame(imputer.fit_transform(game_X), columns=selected_features)
    game_df['model_wp'] = model.predict(game_X_imputed)

    # Normalize actual WP in game_df to home team's perspective
    game_df['wp_normalized'] = np.where(
        game_df['posteam'] == game_df['home_team'],
        game_df['wp'],
        1 - game_df['wp']
    )


    game_mae = mean_absolute_error(game_df['wp'], game_df['model_wp'])

    # Store everything in session_state
    # Update the selected years in session state only after submit
    st.session_state["selected_year"] = [temp_selected_year] 
    st.session_state['model'] = model
    st.session_state['mse'] = mse
    st.session_state['game_df'] = game_df
    st.session_state['game_mae'] = game_mae
    st.session_state['home_team'] = home_team
    st.session_state['away_team'] = away_team
    st.session_state['game_date'] = game_date
    st.session_state['feature_importances'] = model.feature_importances_
    st.session_state['feature_labels'] = selected_features
    st.session_state['game_week'] = game_week
    st.session_state['home_logo'] = home_logo
    st.session_state['away_logo'] = away_logo
    st.session_state['formatted_game_date'] = formatted_game_date
    st.session_state['home_team_full'] = abbr_to_name[home_team]
    st.session_state['away_team_full'] = abbr_to_name[away_team]
    st.session_state['home_rgba'] = home_rgba
    st.session_state['away_rgba'] = away_rgba




elif 'model' in st.session_state and 'game_df' in st.session_state:
    home_team_full = st.session_state.get('home_team_full', home_team)
    away_team_full = st.session_state.get('away_team_full', away_team)
    game_date = st.session_state['game_date']
    model = st.session_state['model']
    mse = st.session_state['mse']
    game_df = st.session_state['game_df']
    game_mae = st.session_state['game_mae']
    home_team = st.session_state['home_team']
    away_team = st.session_state['away_team']

    game_week = st.session_state.get('game_week', 'N/A')
    home_logo = st.session_state.get('home_logo', '')
    away_logo = st.session_state.get('away_logo', '')
    formatted_game_date = st.session_state.get('formatted_game_date')
    home_rgba = st.session_state.get('home_rgba', 'rgba(0,0,255,0.08)')
    away_rgba = st.session_state.get('away_rgba', 'rgba(255,0,0,0.08)')

    st.info("Showing results from your last submitted configuration. Click 'Submit Changes' to update.")

home_team_full = st.session_state.get('home_team_full', home_team)
away_team_full = st.session_state.get('away_team_full', away_team)
home_logo = st.session_state.get('home_logo', '')
away_logo = st.session_state.get('away_logo', '')
game_week = st.session_state.get('game_week', 'N/A')
formatted_game_date = st.session_state.get('formatted_game_date', '')

# Only continue if there's valid data
if 'model' in st.session_state and 'game_df' in st.session_state:
    
    # Game Summary
    home_score = int(round(game_df["home_team_score"].iloc[-1]))
    away_score = int(round(game_df["away_team_score"].iloc[-1]))
    formatted_date = pd.to_datetime(game_date).strftime('%A, %b %d, %Y')

    def format_record(record_str, team_score, opponent_score):
        wins, losses = map(int, record_str.split("–"))

        arrow_style = (
            "display:inline-block; "
            "transform: scale(1.5, 2); "
            "font-weight: normal; "
            "vertical-align: bottom;"
        )

        spacer = (
            "<span style='display:inline-block; "
            "transform: scale(1.6, 2.2); "
            "font-weight: normal; "
            "vertical-align: bottom; "
            "visibility: hidden; "
            "margin: 0 5px;'>⇃</span>"
        )

        if team_score > opponent_score:
            arrow = f"<span style='{arrow_style}; color: #21ba45;'>↾</span>"
            color = "#21ba45"
            formatted = f"{arrow}<span style='margin-left: 6px; font-weight: bold;'>{wins}–{losses}</span>{spacer}"

        elif team_score < opponent_score:
            arrow = f"<span style='{arrow_style}; color: #db2828;'>⇃</span>"
            color = "#db2828"
            formatted = f"{spacer}<span style='margin-right: 6px; font-weight: bold;'>{wins}–{losses}</span>{arrow}"

        else:
            arrow = ""
            color = "#f2c037"
            formatted = f"<span style='font-weight: bold;'>{wins}–{losses}</span>"

        return f"<span style='color: {color}; font-size: 16px;'>{formatted}</span>"


    # Row 1: Team info and score
    col1, col2, col3 = st.columns([1, 1, 1])
    home_team_color = hex_to_color(home_team_color, alpha=1, as_string=True)
    away_team_color = hex_to_color(away_team_color, alpha=1, as_string=True)

    with col1:
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="{home_logo}" style="width: 110px; margin-bottom: 6px;" />
                <div style="font-size: 18px; font-style: italic;">
                    <span style="color: white; text-shadow:
                        -1px -1px 0 {home_team_color},
                        1px -1px 0 {home_team_color},
                        -1px  1px 0 {home_team_color},
                        1px  1px 0 {home_team_color},
                        0px  1px 0 {home_team_color},
                        0px  -1px 0 {home_team_color};">
                        {home_team_full}
                    </span>
                </div>
                <div style="font-size: 42px; font-weight: 900; margin-top: 2px;">{home_score}</div>
                <div style='margin-top: 4px;'>{format_record(home_record, home_score, away_score)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding-top: 105px;'>
                <div style='font-size: 26px; font-weight: bold;'>vs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="{away_logo}" style="width: 110px; margin-bottom: 6px;" />
                <div style="font-size: 18px; font-style: italic;">
                    <span style="color: white; text-shadow:
                        -1px -1px 0 {away_team_color},
                        1px -1px 0 {away_team_color},
                        -1px  1px 0 {away_team_color},
                        1px  1px 0 {away_team_color},
                        0px  1px 0 {away_team_color},
                        0px  -1px 0 {away_team_color};">
                        {away_team_full}
                    </span>
                </div>
                <div style="font-size: 42px; font-weight: 900; margin-top: 2px;">{away_score}</div>
                <div style='margin-top: 4px;'>{format_record(away_record, away_score, home_score)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Row 2: Game Info
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 12px;'>
            <div style='font-size: 16px; font-weight: 500;'>Week {game_week} — {formatted_date}</div>
            <div style='font-size: 14px; color: #CCCCCC;'>@ {venue}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plot Helpers

    def add_possessions(fig, df_sorted):
        df_sorted['possession_change'] = df_sorted['posteam'] != df_sorted['posteam'].shift(1)
        df_sorted.loc[0, 'possession_change'] = True  # Ensure the first row starts a possession

        previous_time = df_sorted['game_seconds_remaining'].iloc[0]
        previous_posteam = df_sorted['posteam'].iloc[0]

        for i, row in df_sorted.iterrows():
            if row['possession_change'] and i != 0:
                current_time = row['game_seconds_remaining']
                team_side = 'home' if previous_posteam == row['home_team'] else 'away'
                fillcolor = home_rgba if team_side == 'home' else away_rgba

                # Shaded region for possession
                fig.add_shape(
                    type="rect",
                    xref="x", yref="paper",
                    x0=current_time, x1=previous_time,
                    y0=0, y1=1,
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    layer='below'
                )

                # Translucent green dotted line for possession change
                fig.add_shape(
                    type="line",
                    xref="x", yref="y",
                    x0=current_time, x1=current_time,
                    y0=0, y1=100,
                    line=dict(
                        color="rgba(0, 255, 0, 0.2)",
                        width=1,
                        dash="dash"
                    ),
                    layer='above'
                )

                previous_time = current_time
                previous_posteam = row['posteam']

        final_time = df_sorted['game_seconds_remaining'].iloc[-1]
        team_side = 'home' if previous_posteam == df_sorted['home_team'].iloc[0] else 'away'
        fillcolor = home_rgba if team_side == 'home' else away_rgba

        return fig


    
    tick_vals = list(range(3600, -1, -900))
    tick_labels = [f"{t//60}m" for t in tick_vals]

    def generate_wp_plot(df, wp_col, title, line_color='blue'):
        df_sorted = df.sort_values(by='game_seconds_remaining', ascending=False)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_sorted['game_seconds_remaining'],
            y=df_sorted[wp_col] * 100,
            mode='lines',
            name=title,
            line=dict(color=line_color, width=3),
            hoverinfo='text',
            hovertext=[
                f"Time: {td}<br>WP: {wp:.2f}%<br>"
                f"{home_team} Score: {hs}<br>{away_team} Score: {as_}"
                for td, wp, hs, as_ in zip(
                    df_sorted['time_display'], 
                    df_sorted[wp_col] * 100,
                    df_sorted['home_team_score'],
                    df_sorted['away_team_score']
                )
            ]
        ))

        fig = add_possessions(fig, df_sorted)

        fig.add_layout_image(dict(
            source=home_logo,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            sizex=0.15, sizey=0.15,
            xanchor="right", yanchor="top",
            layer="above"
        ))
        fig.add_layout_image(dict(
            source=away_logo,
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            sizex=0.15, sizey=0.15,
            xanchor="right", yanchor="bottom",
            layer="above"
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time Remaining (sec)",
            yaxis_title="Win Probability (%)",
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_labels,
                range=[3600, 0]
            ),
            yaxis=dict(range=[0, 100]),
            height=400,
            hovermode='closest',
            showlegend=False
        )
        return fig

    def generate_combined_plot(df):
        df_sorted = df.sort_values(by='game_seconds_remaining', ascending=False)
        fig = go.Figure()

        # Actual WP
        fig.add_trace(go.Scatter(
            x=df_sorted['game_seconds_remaining'],
            y=df_sorted['wp_normalized'] * 100,
            mode='lines',
            name='Actual WP',
            line=dict(color='blue', width=3),
            hoverinfo='text',
            hovertext=[
                f"Time: {td}<br>Actual WP: {wp:.2f}%<br>"
                f"{home_team} Score: {hs}<br>{away_team} Score: {as_}"
                for td, wp, hs, as_ in zip(
                    df_sorted['time_display'],
                    df_sorted['wp_normalized'] * 100,
                    df_sorted['home_team_score'],
                    df_sorted['away_team_score']
                )
            ]
        ))

        # Model WP
        fig.add_trace(go.Scatter(
            x=df_sorted['game_seconds_remaining'],
            y=df_sorted['model_wp'] * 100,
            mode='lines',
            name='Model WP',
            line=dict(color='firebrick', width=3),
            hoverinfo='text',
            hovertext=[
                f"Time: {td}<br>Model WP: {wp:.2f}%<br>"
                f"{home_team} Score: {hs}<br>{away_team} Score: {as_}"
                for td, wp, hs, as_ in zip(
                    df_sorted['time_display'],
                    df_sorted['model_wp'] * 100,
                    df_sorted['home_team_score'],
                    df_sorted['away_team_score']
                )
            ]
        ))

        fig = add_possessions(fig, df_sorted)

        fig.add_layout_image(dict(
            source=home_logo,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            sizex=0.15, sizey=0.15,
            xanchor="right", yanchor="top",
            layer="above"
        ))
        fig.add_layout_image(dict(
            source=away_logo,
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            sizex=0.15, sizey=0.15,
            xanchor="right", yanchor="bottom",
            layer="above"
        ))


        fig.update_layout(
            title=f"Combined Win Probability",
            xaxis_title="Time Remaining (sec)",
            yaxis_title="Win Probability (%)",
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_labels,
                range=[3600, 0]
            ),
            yaxis=dict(range=[0, 100]),
            height=400,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05
            ),
            margin=dict(r=160)
        )
        return fig


    def generate_delta_plot(df):
        df_sorted = df.sort_values(by='game_seconds_remaining', ascending=False)
        df_sorted['delta'] = (df_sorted['model_wp'] - df_sorted['wp_normalized']) * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_sorted['game_seconds_remaining'],
            y=df_sorted['delta'],
            mode='lines',
            name='Delta',
            line=dict(color='purple', width=2),
            hoverinfo='text',
            hovertext=[
                f"Time: {td}<br>"
                f"Model WP: {mwp:.2f}%<br>"
                f"Actual WP: {awp:.2f}%<br>"
                f"Delta: {(mwp - awp):+.2f} pts"
                for td, mwp, awp in zip(
            df_sorted['time_display'],
            df_sorted['model_wp'] * 100,
            df_sorted['wp_normalized'] * 100
                )
            ]
        ))

        fig.update_layout(
            title='Model vs Actual WP Delta (percentage points)',
            xaxis_title='Time Remaining (sec)',
            yaxis_title='Delta (Model - Actual)',
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_labels,
                range=[3600, 0]
            ),
            height=400,
            hovermode='closest'
        )
        return fig

    # Show All Plots
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            generate_wp_plot(game_df, 'model_wp', f"Predicted Win Probability", line_color='firebrick'),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            generate_wp_plot(game_df, 'wp_normalized', f"Actual Win Probability", line_color='blue'),
            use_container_width=True
        )

    st.plotly_chart(generate_combined_plot(game_df), use_container_width=True)
    st.plotly_chart(generate_delta_plot(game_df), use_container_width=True)

    # Show Metrics
    col_mse, col_mae = st.columns(2)

    with col_mse:
        st.metric(label="Model MSE (All Data)", value=f"{mse:.4f}")

    with col_mae:
        st.metric(label="Selected Game MAE", value=f"{game_mae:.4f}")

    # Feature Importance Plot (using cached features)
    if 'feature_importances' in st.session_state and 'feature_labels' in st.session_state:
        cached_importances = st.session_state['feature_importances']
        cached_features = st.session_state['feature_labels']

        fi_df = pd.DataFrame({
            'Feature': cached_features,
            'Importance': cached_importances
        })

        inverse_mapping = {v: k for k, v in feature_mapping.items()}
        fi_df['Feature'] = fi_df['Feature'].map(inverse_mapping)

        fi_fig = px.bar(
            fi_df, 
            x='Feature', 
            y='Importance',
            title="Feature Importance",
            labels={"Feature": "Feature", "Importance": "Importance"},
            log_y=True
        )
        fi_fig.update_layout(
            yaxis=dict(showgrid=False, showticklabels=False),
            xaxis_title=None,
        )
        st.plotly_chart(fi_fig, use_container_width=True)

    # Download CSV
    @st.cache_data
    def convert_df_to_csv(df_):
        return df_.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(game_df)
    st.download_button(
        label="Download Game Predictions as CSV",
        data=csv_data,
        file_name='game_predictions.csv',
        mime='text/csv'
    )