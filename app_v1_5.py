from dash import dcc, html, dash_table, Input, Output, callback, ALL, dash_table
import dash_bootstrap_components as dbc # dash app layout (package name: dash-bootstrap-components)
import plotly.graph_objects as go # plots
from dotenv import load_dotenv # .env file access
import plotly.express as px # plots
from io import BytesIO # handle reading files
import pandas as pd
import boto3 # aws s3 access
import dash
import os

#################### SETUP (GLOBAL) ####################
### Initialise Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "NBA Parlay Dashboard"

### Setup AWS S3 Bucket Access
load_dotenv(f"{os.getcwd()}/.env") # set to directory .env file
bucket_name = os.getenv("AWS_BUCKET_NAME")
access_key = os.getenv("AWS_ACCESS_KEY")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")
s3 = boto3.client("s3",
    aws_access_key_id = access_key,
    aws_secret_access_key = secret_access_key,
    region_name = region
)

### Initialise Datasets from AWS S3 Bucket
def get_dataset_aws(season, entity_type):
    boxscore_regular = s3.get_object(Bucket=bucket_name, Key=f"{season}_{entity_type.lower()}s_boxscore_regular.csv")
    boxscore_regular = pd.read_csv(BytesIO(boxscore_regular['Body'].read()))
    entity_type_boxscore = None # placeholder
    try:
        boxscore_playoffs = s3.get_object(Bucket=bucket_name, Key=f"{season}_{entity_type.lower()}s_boxscore_playoffs.csv")
        boxscore_playoffs = pd.read_csv(BytesIO(boxscore_playoffs['Body'].read()))
        entity_type_boxscore = pd.concat([boxscore_regular, boxscore_playoffs])
    except s3.exceptions.NoSuchKey: # file does not exist in given bucket
        entity_type_boxscore = boxscore_regular
    return entity_type_boxscore.sort_values('GAME DATE') # oldest to newest (ascending)
season = "2024-25"
entity_type_datasets_dict = {
    "PLAYER": get_dataset_aws(season, entity_type="PLAYER"),
    "TEAM": get_dataset_aws(season, entity_type="TEAM")
}

### Initialise Team Names
nba_teams = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

### Get Entity Dashboard Title
def get_dashboard_title_string(entity_name, entity_type, team_name):
    if entity_type == 'PLAYER':
        return f"{entity_name} ({nba_teams[team_name]})"
    else:
        return f"{nba_teams[entity_name]} ({entity_name})"
    
### Get Entity Props Sliders
def get_entity_props_sliders(df, props):
    sliders = []
    for prop in props:
        prop_col = df[prop]
        min_val, max_val, avg = prop_col.min(), prop_col.max(), round(prop_col.mean())
        slider = dcc.Slider(
            id = {"type": "prop-slider", "index": prop}, # IMPORTANT!!!!
            min = 0, max = max_val, value = avg,
            step = 1,
            marks = None, # marks = {min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
        # sliders.append( html.Div([html.Label(f'{prop}:'), slider]) )
        sliders.append(
            dbc.Row([
                dbc.Col([html.Label(f'{prop}:', style={"fontWeight": "bold"})], width='auto'),
                dbc.Col([slider])
            ])
        )
    return sliders # list of html.Div's containing a slider each

### Get Table of Counts for Multiplicative Odds Calculation
def get_entity_props_counts_detailed(prop_counts_df):
    total = prop_counts_df['COUNT'].sum()
    # prop_counts_df['PDF'] = prop_counts_df['COUNT'] / total
    cum_sum_count = prop_counts_df['COUNT'].cumsum()
    prop_counts_df['CDF'] = cum_sum_count / total # UNDER's: P(X <= x) = F(X = x)
    prop_counts_df['CDF x-1'] = prop_counts_df['CDF'].shift(1, fill_value=0) # shift 'CDF' down one cell [i.e. P(X <= x-1) = F(X = x-1)]
    prop_counts_df['REVERSE CDF'] = 1 - prop_counts_df['CDF x-1'] # OVER's: P(X >= x) = 1 - P(X < x) = 1 - P(X <= (x-1)) = 1 - F(X = (x-1))
    return prop_counts_df
def get_entity_props_counts(df, props, entity_type):
    # Pivot and Count Prop Occurrences
    long_df = pd.melt(df, id_vars=entity_type, value_vars=props, var_name='PROP', value_name='VALUE') # long format df
    prop_value_occurrence_count = long_df.groupby(['PROP','VALUE']).apply(lambda x: x.shape[0], include_groups=False).reset_index() # for every prop+value combination get count
    prop_value_occurrence_count = prop_value_occurrence_count.rename(columns={0: "COUNT"})
    # Attach Prop Occurences CDF (UNDER Probability), Reverse CDF (OVER Probability)
    final_counts = prop_value_occurrence_count.groupby('PROP').apply(get_entity_props_counts_detailed, include_groups=False).reset_index(level=0)
    return final_counts# Note: still grouped by PROP

### Get DataTable of Prop Slider Odds
def get_probs_datatable(odds_df):
    dt = dash_table.DataTable(
        data = odds_df.to_dict('records'),
        columns = [{"name": col, "id": col} for col in odds_df.columns],
        style_data_conditional = [
            {
                'if': {'column_id': 'Probability', 'row_index': i},
                'backgroundColor': get_prob_dt_cell_colour(odds_df['Probability'].iloc[i]) # issue before - background colour has to be singular not multiple (if applying to whole or singular column)
            } for i in range(odds_df.shape[0])
        ]
    )
    return dt
def get_prob_dt_cell_colour(prob):
    idx = int(prob * 10)
    return px.colors.diverging.RdYlGn[idx]

## Cards for Entity Dashboard
def get_cards(df, props):
    if len(props) > 1: # put combination prop first
        props = ["+".join(props)] + props
    cards = []
    for prop in props:
        prop_col = df[prop]
        card = dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H2(f"{prop_col.mean(): .2f}"),
                    html.P(f"Average {prop} (SD: {prop_col.std(): .2f})", style={'font-weight': 'bold'}),
                    html.P(f"Last 5: {prop_col.iloc[-5:].mean(): .2f} (SD: {prop_col.iloc[-5:].std(): .2f})", style={'font-size': '0.9em'}) # reduce size to 90% of default
                ])
            )
        ], width='auto', className='pb-3')
        cards.append(card)
    return cards

### Timeseries barplot for given Prop
def get_z_score(df_col):
    mean = df_col.mean()
    sd = df_col.std()
    z_score = (df_col - mean) / sd
    # shift and scale (0-1)
    min_val, max_val = 5, -5 # Note: VALUES ARE SWAPPED  SINCE (1 = BLUE, 0 = RED) ON COLOUR SCALE
    adj_z_score = (z_score - min_val) / (max_val - min_val) # convert to 0-1 scale based on -5/5 values -- on N(0,1) distribution, -5 or 5 have < 0.1% chance so adjust scale by +(-5)
    return round(adj_z_score, 4)
def get_cell_colour(z_score): # categorise z_score to value between 0-1 (index 0-10) then assign color
    if pd.isna(z_score): # NaN from no attempts - colour as blue (cold) as possible
        return px.colors.diverging.RdBu[9] # not using 10 as its hard to see
    else:
        colourscale_idx = int(round(z_score, 1) * 10) # e.g. round(z_score, 1) to nearest tenth
        return px.colors.diverging.RdBu[colourscale_idx]
def get_bar_ts(df, props):
    df['H/A'] = ["vs." if val == 1 else "@" for val in df['HOME']]
    prop_name = "+".join(props)
    colors = [get_cell_colour(z_score) for z_score in get_z_score(df[prop_name])] # calculate z_scores for column then get colour
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x = df['GAME DATE'], y = df[prop_name],
            marker_color = colors, textposition = 'auto', texttemplate = '%{y}',
            hovertext = df.loc[:, ["TEAM OPP", "H/A", prop_name]].astype(str),
            hovertemplate = "<br>".join([
                '%{x}',
                '%{hovertext[1]} %{hovertext[0]}',
                '%{y}'
            ]),
            showlegend = False,
            name = prop_name # name the trace
        )
    )
    fig = fig.update_layout(
        title_text=f"{prop_name} Over Time",
        xaxis=dict(title="GAME DATE", rangeslider=dict(visible=True), type="date"), # adds date range slider to plot
        yaxis=dict(title=prop_name)
    )
    return fig

## Histogram for Entity Prop
def get_histogram(df, prop):
    prop_name = "+".join(prop)
    hist_title = f'{prop_name} Distribution'
    fig = px.histogram(df, x=prop_name, histnorm='probability', marginal='box', title=hist_title)
    return fig

## Averages Line Chart
def add_scatter_line(x, y, name, showlegend, mode, line):
    return go.Scatter(x=x, y=y, name=name, showlegend=showlegend, mode=mode, line=line)
def get_rolling_line(df, prop, N):
    prop_name = "+".join(prop)
    rolling_avg = df[prop_name].rolling(N).mean()
    rolling_std = df[prop_name].rolling(N).std()
    avg_lower = rolling_avg - rolling_std
    avg_higher = rolling_avg + rolling_std
    fig = go.Figure()
    fig.add_trace(add_scatter_line(x=df['GAME DATE'], y=rolling_avg, name=f'{prop_name} AVG. {N}', showlegend=True, mode='lines', line=dict(dash='solid')))
    fig.add_trace(add_scatter_line(x=df['GAME DATE'], y=avg_lower, name=f'{prop_name} AVG. {N} LOWER', showlegend=False, mode='lines', line=dict(dash='dot', color='red')))
    fig.add_trace(add_scatter_line(x=df['GAME DATE'], y=avg_higher, name=f'{prop_name} AVG. {N} UPPER', showlegend=False, mode='lines', line=dict(dash='dot', color='red')))
    fig.update_layout(
        hovermode = "x",
        title_text = f'{prop_name} {N}-Game Average',
        showlegend = False
    )
    return fig

### CDF Plot for Given Player Prop
def get_cdf_line(df, prop, complement):
    prop_name = "+".join(prop)
    over_under_type = ['UNDER', 'OVER'][complement]
    cdf_title = f"{prop_name} {over_under_type} Probability"
    fig = None
    if complement is True: # Complement CDF = P(X > x) = 1 - P(X <= x) <<i.e. Players OVER Probabilities >>
        fig = px.ecdf(df, x=prop_name, ecdfmode='complementary', title=cdf_title)
    else: # CDF = P(X <= x) <<i.e. Players UNDER Probabilities >>
        fig = px.ecdf(df, x=prop_name, title=cdf_title)
    fig.update_layout(hovermode="x")
    return fig

#################### LAYOUT (UI) ####################
app.layout = html.Div([
    dbc.Row([
        html.H1("NBA Parlay Probability Calculator", style={'color': 'white'})
    ], style={'background-color': 'black'}, className='px-2 py-2'),
    ### Upper Section
    dbc.Row([
        ### Left Side: User Inputs
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H5("Entity Type:", style={"fontWeight": "bold"})
                ], width='auto'),
                dbc.Col([
                    dcc.RadioItems(
                        id = 'entity_type_radio',
                        options = [
                            {'label': 'Player', 'value': 'PLAYER'},
                            {'label': 'Team', 'value': 'TEAM'}
                        ],
                        value = 'PLAYER',
                        inline = True,
                        labelStyle={'margin-right': '15px'}
                    )
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Prop(s):", style={"fontWeight": "bold"})
                ], width='auto'),
                dbc.Col([
                    dcc.Dropdown(
                        id = "props_dd",
                        options = [],
                        value = [],
                        multi = True
                    )
                ])
            ], className='pb-2'),
            dbc.Row([
                dbc.Col([
                    html.H5("Entity:", style={"fontWeight": "bold"})
                ], width='auto'),
                dbc.Col([
                    dcc.Dropdown(
                        id = "entity_dd",
                        options = [],
                        value = None,
                        multi = False
                    )
                ])
            ])
        ], width=6),
        ### Right Side: Parlay DataTable
        dbc.Col(id='parlay_datatable', width=6)
    ], className='px-2 py-2'),
    html.Hr(),

    dcc.Store(id='entity_type_df'), # switches between TEAM and PLAYER datasets appropriately
    dcc.Store(id='entity_df'), # subset of 'entity_type_df' (entity specific)
    dcc.Store(id='active_parlay_df'),
    dcc.Store(id='entity_prop_counts_df'),

    ### MAIN DASHBOARD SECTION
    dbc.Row(id='entity_dashboard_title', className='px-2'),
    dbc.Row([
        ### Left Side: Prop Inputs & Odds
        dbc.Col(id='entity_prop_inputs', width=3),
        ### Right Side: Main Plots
        dbc.Col([
            dbc.Row(id='entity_prop_cards', className='pb-0'),
            dbc.Row([
                dbc.Col(id='entity_prop_bar', width=6), # ***** up to here *****
                dbc.Col(id='entity_prop_hist', width=6)
            ]),
            dbc.Row([
                dbc.Col(id='entity_prop_avg5', width=4),
                dbc.Col(id='entity_prop_under_cdf', width=4),
                dbc.Col(id='entity_prop_over_cdf', width=4)
            ]),
            dbc.Row([
                dbc.Col(id='entity_prop_boxscore')
            ])
        ], width=9)
    ], className='px-2 py-2')
])

#################### SERVER ####################
## [DISPLAY] Active Entity Boxscore DataTable
@callback(
        Output('entity_prop_boxscore', 'children'),
        [Input('entity_df', 'data'),
         Input('props_dd', 'value'),
         Input('entity_type_radio', 'value')]
)
def display_entity_prop_boxscore(entity_dict, props, entity_type):
    if entity_dict != {} and props != []:
        df = pd.DataFrame(entity_dict).iloc[::-1] # latest games first
        if len(props) > 1:
            props = props + ["+".join(props)]
        # Get Cell Colours for Prop Columns (Needs to be standardised to WHOLE column - not just rows shown)
        colors = {}
        for prop in props:
            col_colours = [get_cell_colour(z_score) for z_score in get_z_score(df[prop])] # calculate z_scores for column then get colour
            colors[prop] = col_colours
        # DataTable
        select_cols = ['TEAM','GAME DATE','TEAM OPP','MIN'] + props + ['WIN','HOME']
        if entity_type == 'PLAYER': 
            select_cols = ['PLAYER'] + select_cols
        subset_df = df[select_cols] # subset columns of entity dataframe (plus re-orders)
        dt = dash_table.DataTable(
            data=subset_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in subset_df.columns],
            style_data_conditional = [ # color probability column
                {
                    'if': {'column_id': prop, 'row_index': i},
                    'backgroundColor': colors[prop][i]
                } for prop in props for i in subset_df.index
            ],
            style_cell={'textAlign': 'left'},
            filter_action="native"
        )
        return [
            html.H4("Previous Games"), 
            dt
        ]
    else:
        return []

### [DISPLAY] Entity Prop's Timeseries Barplot, Histogram Distribution, Rolling Average Plot, CDF Plot
@callback(
	[Output('entity_prop_bar', 'children'),
    Output('entity_prop_hist', 'children'),
    Output('entity_prop_avg5', 'children'),
    Output('entity_prop_under_cdf', 'children'),
    Output('entity_prop_over_cdf', 'children')],
	[Input('entity_df', 'data'),
	Input('props_dd', 'value'),
	Input('entity_dd', 'value'),
    Input('entity_type_radio', 'value')]
)
def create_plots(entity_dict, props, entity, entity_type):
    if props != [] and entity_dict != {}:
        df = pd.DataFrame(entity_dict)
        ts_bar = get_bar_ts(df, props) # Entity Prop Time Series Barplot
        props_hist = get_histogram(df, props) # Entity Prop Histogram (Distribution)
        ts_l5_avg = get_rolling_line(df, props, 5) # Entity Prop Rolling 5-Average
        under_cdf = get_cdf_line(df, props, False) # Entity Prop CDF (UNDER) Line Plot
        over_cdf = get_cdf_line(df, props, True) # Entity Prop CDF (OVER) Line Plot
        return ([dcc.Graph(figure=ts_bar)], [dcc.Graph(figure=props_hist)], 
                [dcc.Graph(figure=ts_l5_avg)], [dcc.Graph(figure=under_cdf)], [dcc.Graph(figure=over_cdf)])
    else:
        return [], [], [], [], []

### [DISPLAY] Card Averages for each Entity Prop
@callback(
        Output('entity_prop_cards', 'children'),
        [Input('entity_df', 'data'),
         Input('props_dd', 'value')]
)
def create_card_averages(entity_dict, props):
    if props != [] and entity_dict != {}:
        df = pd.DataFrame(entity_dict)
        cards = get_cards(df, props) # list of dbc.Col() containing a dbc.Card() for each prop
        return cards
    else:
        return []

### [UPDATE] Reset Parlay DataFrame When Clicked
@callback(
        [Output('active_parlay_df', 'data', allow_duplicate=True),
         Output('reset_props_list_bttn', 'n_clicks')],
        Input('reset_props_list_bttn', 'n_clicks'),
        prevent_initial_call=True
)
def reset_props_list_df(n_clicks):
    if n_clicks > 0:
        return None, 0
    
### [DISPLAY] Parlay DataFrame
@callback(
        Output('parlay_datatable', 'children'),
        Input('active_parlay_df', 'data')
)
def display_parlay_datatable(parlay_dict):
    if parlay_dict is None: # nothing stored yet
        temp = pd.DataFrame({'ENTITY': [], 'PROP': [], 'O/U': [], 'VALUE': [], 'PROBABILITY': [], 'PERCENTAGE': []})
        dt =  dash_table.DataTable(data=temp.to_dict('records'), columns=[{"name": i, "id": i} for i in temp.columns], style_cell={'textAlign': 'left'})
        return [html.H3("Parlay Probability"), dt]
    else:
        # Process Dataset
        df = pd.DataFrame(parlay_dict)
        df = df.groupby(['ENTITY','PROP']).last().reset_index() # returns last (latest) row for every group
        df.loc[df.shape[0]] = ['TOTAL PROBABILITY', '', '', None, df['PROBABILITY'].product()]
        df['PROBABILITY'] = round(df['PROBABILITY'], 6) # rounding only for output purposes (not rounded in calculations)
        df['PERCENTAGE'] = round(df['PROBABILITY'] * 100, 2)
        # Display Props List Table
        dt = dash_table.DataTable( # more at: https://dash.plotly.com/datatable/interactivity
            id='prop_list_datatable',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_data_conditional = [ # color probability column
                {
                    'if': {'column_id': 'PROBABILITY', 'row_index': i},
                    'backgroundColor': get_prob_dt_cell_colour(df['PROBABILITY'].iloc[i]) # issue before - background colour has to be singular not multiple (if applying to whole or singular column)
                } for i in range(df.shape[0])
            ],
            style_cell={'textAlign': 'left'},
            # row_selectable="multi",
            # selected_rows=[]
        )
        reset_bttn = html.Button('Clear All', id='reset_props_list_bttn', n_clicks=0)
        return [html.H3("Parlay Probability"), dt, reset_bttn]

### [STORE ACTIVE] Props List (Parlay) Dataframe
@callback(
        [Output('active_parlay_df', 'data'),
         Output('sliders_bttn', 'n_clicks')],
        [Input('active_parlay_df', 'data'),
        Input('entity_prop_counts_df', 'data'),
        Input({"type": "prop-slider", "index": ALL}, 'value'), # ALL things with the id {'type': 'prop-slider'} are gathered here!!!
        Input('sliders_bttn', 'n_clicks'),
        Input('props_dd', 'value'),
        Input('over_under_radio', 'value'),
        Input('entity_dd', 'value')]
)
def store_props_list_df(parlay_dict, prop_counts_dict, slider_values, n_clicks, props, over_under, entity):
    if n_clicks > 0: # Note: button only appears once props and entity inputs are given (so current entity data should be stored already)
        # Get each slider inputs corresponding probability
        counts_df = pd.DataFrame(prop_counts_dict)
        prop_probs = []
        cdf_types = {'OVER': 'REVERSE CDF', 'UNDER': 'CDF'}
        for i, prop in enumerate(props):
            slider_value = slider_values[i]
            cdf_type = cdf_types[over_under]
            prop_counts = counts_df.groupby('PROP').get_group(prop)
            prop_prob = None # temporary
            i = prop_counts['VALUE'].searchsorted(slider_value, side="left")
            if over_under == "UNDER": # i.e. CDF = P(X <= x) <=> F(X = x)
                if (i == 0) and (slider_value < prop_counts['VALUE'].iloc[i]): # slider less than minimum observed
                    prop_prob = 0 # dealing with UNDER (CDF)
                else:
                    prop_prob = prop_counts[cdf_type].iloc[i]
            else: # "OVER" (Reverse CDF) = P(X >= x) = 1 - P(X < x) <=> 1 - P(X <= (x-1)) <=> 1 - F(X = (x-1))
                if i == prop_counts.shape[0]: # slider more than maximum observed
                    prop_prob = 0 # dealing with OVER (Reverse CDF)
                else:
                    prop_prob = prop_counts[cdf_type].iloc[i]
            prop_probs.append( [entity, prop, over_under, slider_value, prop_prob] )
        entity_parlay_df = pd.DataFrame(prop_probs, columns=['ENTITY','PROP','O/U','VALUE','PROBABILITY'])
        # Append to existing data and Store as json
        new_parlay_df = entity_parlay_df if parlay_dict is None else pd.concat([pd.DataFrame(parlay_dict), entity_parlay_df])
        return new_parlay_df.to_dict('records'), 0 # only add to current_output if clicked + always reset button state (regardless if pressed or not)

### [DISLAY] DataTable of Current Entity Props Inputs Probabilitities
@callback(
    Output('entity_prop_odds_section', 'children'),
    [Input('entity_prop_counts_df', 'data'),
    Input({"type": "prop-slider", "index": ALL}, 'value'), # ALL things with the id {'type': 'prop-slider'} are gathered here!!!
    Input('props_dd', 'value'),
    Input('entity_dd', 'value'),
    Input('over_under_radio', 'value')]
)
def display_entity_odds_table(prop_counts_dict, sliders, props, entity, over_under):
    if (props != []) and (entity != None) and (sliders != []):
        # Get each slider inputs corressponding probability
        counts_df = pd.DataFrame(prop_counts_dict)
        prop_probs, multi_odds = {}, 1
        sign_type = {"OVER": ">=", "UNDER": "<="}
        cdf_types = {'OVER': 'REVERSE CDF', 'UNDER': 'CDF'}
        for i, prop in enumerate(props):
            slider_value = sliders[i]
            sign, cdf_type = sign_type[over_under], cdf_types[over_under]
            prop_counts = counts_df.groupby('PROP').get_group(prop)
            prop_prob = None # temporary
            i = prop_counts['VALUE'].searchsorted(slider_value, side="left")
            if over_under == "UNDER": # i.e. CDF = P(X <= x) <=> F(X = x)
                if (i == 0) and (slider_value < prop_counts['VALUE'].iloc[i]): # slider less than minimum observed
                    prop_prob = 0 # dealing with UNDER (CDF)
                else:
                    prop_prob = prop_counts[cdf_type].iloc[i]
            else: # "OVER" (Reverse CDF) = P(X >= x) = 1 - P(X < x) <=> 1 - P(X <= (x-1)) <=> 1 - F(X = (x-1))
                if i == prop_counts.shape[0]: # slider more than maximum observed
                    prop_prob = 0 # dealing with OVER (Reverse CDF)
                else:
                    prop_prob = prop_counts[cdf_type].iloc[i]
            prop_probs[f'{prop} {sign} {slider_value}'] = round(prop_prob, 6) # round only for output purposes (not rounded in calculations)
            multi_odds *= prop_prob
        prop_probs['Product'] = round(multi_odds, 6) # round only for output purposes (not rounded in calculations)
        # Display Table to represent Additive/Multiplicative Odds
        odds_df = pd.DataFrame.from_dict(prop_probs, orient='index', columns=['Probability']).reset_index(names='Prop')
        return [
            html.Label("Prop Odds", style={"fontWeight": "bold"}),
            get_probs_datatable(odds_df) # dash_table.DataTable()
        ]
    else:
        return []

### [STORE ACTIVE] Entity's Prop Counts
@callback(
        Output('entity_prop_counts_df', 'data'),
        [Input('entity_df', 'data'),
        Input('entity_type_radio', 'value'),
        Input('props_dd', 'value')]
)
def store_entity_props_counts(entity_dict, entity_type, props):
    if entity_dict != {} and props != []:
        entity_df = pd.DataFrame(entity_dict)
        prop_counts = get_entity_props_counts(entity_df, props, entity_type)
        return prop_counts.to_dict("records")
    else:
        return {}

### [DISPLAY] Entity Prop Inputs Section
@callback(
        Output('entity_prop_inputs', 'children'),
        [Input('entity_df', 'data'),
        Input('props_dd', 'value'),
        Input('entity_dd', 'value')]
)
def generate_entity_prop_inputs(entity_dict, props, entity):
    if (props != []) and (entity != None) and (entity_dict != {}):
        df = pd.DataFrame(entity_dict)
        slider_divs = get_entity_props_sliders(df, props)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("OVER or UNDER:", style={"fontWeight": "bold"})
                ], width='auto'),
                dbc.Col([
                    dcc.RadioItems(
                        id = "over_under_radio",
                        options = ["OVER", "UNDER"],
                        value = "OVER",
                        inline = True
                    )
                ])
            ]),
            html.Div(slider_divs),
            dbc.Row([
                dbc.Col([html.Button('Add To Parlay', id='sliders_bttn', n_clicks=0)], width='auto')
            ]),
            html.Div(id='entity_prop_odds_section')
        ], style={
                "border": "1px solid rgba(0, 0, 0, 0.125)",  # Faint black border
                "border-radius": "0.25rem",                  # Rounded corners
                "box-shadow": "0 1px 3px rgba(0, 0, 0, 0.2)",  # Subtle shadow
                "padding": "10px",                            # Padding inside
            })
    else:
        return []

### [DISPLAY] Dashboard Title
@callback(
        Output('entity_dashboard_title', 'children'),
        [Input('entity_df', 'data'),
        Input('entity_type_radio', 'value'),
        Input('entity_dd', 'value')]
)
def display_dashboard_title(entity_dict, entity_type, entity_name):
    if entity_name != None:
        team_name = entity_dict[0]['TEAM']
        dashboard_title = get_dashboard_title_string(entity_name, entity_type, team_name)
        return [html.H1(dashboard_title)]
    else:
        return []

### [STORE ACTIVE] Entity Dataset
@callback(
    Output('entity_df', 'data'),
    [Input('entity_type_df', 'data'),
    Input('entity_type_radio', 'value'),
    Input('entity_dd', 'value'),
    Input('props_dd', 'value')]
)
def store_active_entity_dataset(entity_type_dict, entity_type, entity, props):
    if entity is None:
        return {} # using dict b/c conversion to dataframe should not throw error
    else:
        df = pd.DataFrame(entity_type_dict)
        entity_df = df.groupby(entity_type).get_group(entity).copy() # when working on a subset, use .copy() to avoid CopyWarning from pandas
        if props != [] and len(props) > 1: # add combined column if multile props
            entity_df["+".join(props)] = entity_df[props].sum(axis=1) # attach row sums
        return entity_df.to_dict("records")

### [UPDATE] Entity Dropdown Based On Active Entity Type Dataset
@callback(
        Output('entity_dd', 'options'),
        [Input('entity_type_df', 'data'),
         Input('entity_type_radio', 'value')]
)
def update_entity_dropdown(entity_type_dict, entity_type):
    entities = [d[entity_type] for d in entity_type_dict] # only need one-column not whole dataframe hence no dataframe conversion
    entities = sorted(list(set(entities))) # sorted arrangement of unique entities
    return entities

### [STORE ACTIVE] Entity Type Dataset
@callback(
    Output('entity_type_df', 'data'),
    Input('entity_type_radio', 'value')
)
def store_active_entity_type_dataset(entity_type):
    return entity_type_datasets_dict[entity_type].to_dict("records")

### [UPDATE] Props Dropdown Based On Entity Radio Items
@callback(
	Output('props_dd', 'options'),
	Input('entity_type_radio', 'value')
)
def update_props_dropdown(entity_type):
    entity_props = {
        'PLAYER': ['PTS', 'REB', 'AST', '3PM', 'STL', 'BLK'],
        'TEAM': ['PTS', '+/-']
    }
    return entity_props[entity_type]



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)