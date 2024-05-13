"""
NBA Props Dash App using boxscore data.
"""
from dash import Dash, html, dcc, Input, Output, callback, ALL, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px # plots
import plotly.graph_objects as go # plots
from dotenv import load_dotenv # .env file access
import boto3 # aws s3 access
from io import BytesIO # handle reading files
import pandas as pd
from math import isnan
import os

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "NBA Parlay Dashboard"

##### SETUP CODE #####
## AWS S3 Bucket Settings
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
## Read CSV file from AWS S3 Bucket - current season (implement seasons option later)
players_regular = s3.get_object(Bucket=bucket_name, Key="2023-24_players_boxscore_regular.csv")
players_regular = pd.read_csv(BytesIO(players_regular['Body'].read()))#.sort_values('GAME DATE').set_index('PLAYER') # oldest to newest
players_playoffs = s3.get_object(Bucket=bucket_name, Key="2023-24_players_boxscore_playoffs.csv")
players_playoffs = pd.read_csv(BytesIO(players_playoffs['Body'].read()))#.sort_values('GAME DATE').set_index('PLAYER') # oldest to newest
players = pd.concat([players_regular, players_playoffs]).sort_values('GAME DATE').set_index('PLAYER') # oldest to newest

teams_regular = s3.get_object(Bucket=bucket_name, Key="2023-24_teams_boxscore_regular.csv")
teams_regular = pd.read_csv(BytesIO(teams_regular['Body'].read()))#.sort_values('GAME DATE').set_index('TEAM')
teams_playoffs = s3.get_object(Bucket=bucket_name, Key="2023-24_teams_boxscore_playoffs.csv")
teams_playoffs = pd.read_csv(BytesIO(teams_playoffs['Body'].read()))#.sort_values('GAME DATE').set_index('TEAM')
teams = pd.concat([teams_regular, teams_playoffs]).sort_values('GAME DATE').set_index('TEAM')
## Team/Player Dropdown Options
player_dd_opts = sorted(players.index.unique())
team_dd_opts = sorted(teams.index.unique())


##### APP FUNCTIONS #####
## Convert team abbreviation to full
def get_team_name(team):
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
    return nba_teams[team]
## Title for Entity Dashboard
def get_dashboard_title(entity_df, entity_name, entity_type):
    if entity_type == 'PLAYER':
        return f"{entity_name} ({get_team_name(entity_df['TEAM'].iloc[-1])})"
    else:
        return f"{get_team_name(entity_name)} ({entity_name})"
## Sliders for Multiplicative Odds
def get_sliders_section(df, props):
    sliders = []
    for prop in props:
        prop_col = df[prop]
        min_val, max_val, avg = prop_col.min(), prop_col.max(), round(prop_col.mean())
        slider = dcc.Slider(
            id = {"type": "prop-slider", "index": prop}, # IMPORTANT!!!!
            min = min_val, max = max_val, value = avg,
            step = 1,
            marks = None, # marks = {min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
        sliders.append( html.Div([html.Label(f'{prop}:'), slider]) )
    return sliders # list of html.Div's containing a slider each
## Get Counts Table for Multiplicative Odds Calculation
def get_entity_props_counts_detailed(prop_counts_df):
    total = prop_counts_df['COUNT'].sum()
    prop_counts_df['PDF'] = prop_counts_df['COUNT'] / total
    cum_sum_count = prop_counts_df['COUNT'].cumsum()
    prop_counts_df['CDF'] = cum_sum_count / total # F(X=x) = P(X <= x) i.e. UNDER's
    prop_counts_df['REVERSE CDF'] = 1 - prop_counts_df['CDF'] # 1 - F(X=x) <=> P(X > x) i.e. OVER's
    return prop_counts_df
def get_entity_props_counts(df, props, entity_type):
    df = df[props].reset_index() # adds entity_type as column
    df = pd.melt(df, id_vars=entity_type, value_vars=props, var_name='PROP', value_name='VALUE') # long format df
    counts = df.groupby([entity_type,'PROP','VALUE']).apply(lambda x: x.shape[0]) # for every prop+value combination get count
    counts = counts.reset_index().rename(columns={0: f'COUNT'})
    # Returns: df with entity_type, prop_type, prop_type_value, count, pdf, cdf, reverse cdf
    return counts.groupby([entity_type,'PROP']).apply(get_entity_props_counts_detailed).reset_index(drop=True)
## Get Slider Input Probabiltities
def get_index_position(prop_col, value):
    last_seen = 0
    for i, num in enumerate(prop_col):
        if value > num:
            last_seen = i
        elif value <= num: # take the floor (last seen num)
            break
    return last_seen
def get_input_probability(df, prop, value, over_under_type):
    df = df.set_index('PROP').loc[[prop]]
    over_under_col = {'OVER': 'REVERSE CDF', 'UNDER': 'CDF'}[over_under_type]    # Get index position in the number range - process is the same for CDF/Reverse CDF
    i = get_index_position(df['VALUE'], value)
    return df[over_under_col].iloc[i]
## Get DataTable of Slider Probabilities
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
        card = dbc.Card(
            dbc.CardBody([
                html.H2(f"{prop_col.mean(): .2f}"),
                html.P(f"Average {prop} (Std.Dev: {prop_col.std(): .2f})", style={'font-weight': 'bold'}),
                html.P(f"Last 5: {prop_col[-5:].mean(): .2f} ({prop_col[-5:].std(): .2f})", style={'font-size': '0.9em'}) # reduce size to 90% of default
            ])
        )
        cards.append(card)
    return cards # list of dbc.Card() for every prop
## Timeseries barplot for given Prop
def get_z_score(df_col):
    mean = df_col.mean()
    sd = df_col.std()
    z_score = (df_col - mean) / sd
    # shift and scale (0-1)
    min_val, max_val = 5, -5 # Note: VALUES ARE SWAPPED  SINCE (1 = BLUE, 0 = RED) ON COLOUR SCALE
    adj_z_score = (z_score - min_val) / (max_val - min_val) # convert to 0-1 scale based on -5/5 values -- on N(0,1) distribution, -5 or 5 have < 0.1% chance so adjust scale by +(-5)
    return round(adj_z_score, 4)
def get_cell_colour(z_score): # categorise z_score to value between 0-1 (index 0-10) then assign color
    if isnan(z_score): # NaN from no attempts - colour as blue (cold) as possible
        return px.colors.diverging.RdBu[9] # not using 10 as its hard to see
    else:
        colourscale_idx = int(round(z_score, 1) * 10) # e.g. round(z_score, 1) to nearest tenth
        return px.colors.diverging.RdBu[colourscale_idx]
def get_bar_ts(df, props, entity_name):
    df['H/A'] = ["vs." if val == 1 else "@" for val in df['HOME']]
    prop_name = "+".join(props)
    colors = [get_cell_colour(z_score) for z_score in get_z_score(df[prop_name])] # calculate z_scores for column then get colour
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x = df['GAME DATE'], y = df[prop_name],
            marker_color = colors, textposition = 'auto', texttemplate = '%{y}',
            hovertext = df.loc[:, ["TEAM OPP","H/A",prop_name]].astype(str),
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
        title_text=f"{entity_name} {prop_name} Over Time",
        xaxis=dict(title="GAME DATE", rangeslider=dict(visible=True), type="date"), # adds date range slider to plot
        yaxis=dict(title=prop_name)
    )
    return fig
## Averages Line Chart
def add_scatter_line(x, y, name, showlegend, mode, line):
    return go.Scatter(x=x, y=y, name=name, showlegend=showlegend, mode=mode, line=line)
def get_rolling_line(df, prop, N, entity_name):
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
        title_text = f'{entity_name} Average {N}-Game {prop_name}',
        showlegend = False
    )
    return fig
## Histogram for Given Prop
def get_histogram(df, prop, entity_name):
    prop_name = "+".join(prop)
    hist_title = f'{entity_name} {prop_name} Distribution'
    fig = px.histogram(df, x=prop_name, histnorm='probability', marginal='box', title=hist_title)
    return fig
## CDF Plot for Given Player Prop
def get_cdf_line(df, prop, complement, entity_type):
    prop_name = "+".join(prop)
    over_under_type = ['UNDER', 'OVER'][complement]
    pname = df[entity_type].iloc[-1]
    cdf_title = f"{pname} {prop_name} {over_under_type} Probability"
    fig = None
    if complement is True: # Complement CDF = P(X > x) = 1 - P(X <= x) <<i.e. Players OVER Probabilities >>
        fig = px.ecdf(df, x=prop_name, ecdfmode='complementary', title=cdf_title)
    else: # CDF = P(X <= x) <<i.e. Players UNDER Probabilities >>
        fig = px.ecdf(df, x=prop_name, title=cdf_title)
    fig.update_layout(hovermode="x")
    return fig


##### LAYOUT #####
# Useful css padding tips: https://getbootstrap.com/docs/4.0/utilities/spacing/ 
app.layout = html.Div([
    dbc.Row([
        html.H1("NBA Parlay Probability Calculator", style={'color': 'white'})
    ], className='px-2 py-2', style={'background-color': 'black'}),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([html.H5("Player or Team:")], width="auto"),
                dbc.Col([
                    dcc.RadioItems(
                        id ="entity_type_radio",
                        options=["PLAYER", "TEAM"],
                        value="PLAYER",
                        inline=True,
                        labelStyle={'margin-right': '15px'}
                    )
                ])
            ], className='pb-1'),
            dbc.Row([
                dbc.Col([html.H5("Prop(s):")], width="auto"),
                dbc.Col([
                    dcc.Dropdown(
                        id="prop_dd",
                        options=[],
                        value=[],
                        multi=True
                    )
                ])
            ], className='pb-2'),
            dbc.Row([
                dbc.Col([html.H5("Name:")], width="auto"),
                dbc.Col([
                    dcc.Dropdown(
                        id="entity_dd",
                        options=[],
                        value=[],
                        multi=False
                    )
                ])
            ]),
        ], width=6),
        dbc.Col(id='parlay_datatable', width=6)
    ], className='px-2 py-2'),

    ## Store Active Dataframe ##
    dcc.Store(id='active_df'), # for more on limitations of dcc.Store: https://community.plotly.com/t/what-is-the-purpose-of-dcc-store-if-plots-are-generated-on-the-backend-with-python/64963/10
    dcc.Store(id='active_parlay_df'),
    html.Hr(),

    dbc.Row(id='entity_dashboard_title', className='px-2 py-2'),
    dbc.Row([
        dbc.Col([ ## Left Side (Cards, Sliders, Prob. Table)
            dbc.Row(id='sliders_section', className='pb-4'),
            dbc.Row(id='odds_table_section')
        ], width=2, className='px-4 py-2'),
        dbc.Col([ ## Right Side (Tab of Plots)
            dbc.Row(id='cards_section', className='py-2'),
            dbc.Row([
                dbc.Col(id='timeseries_bar', width=6, className='py-2'),
                dbc.Col(id='probability_hist', width=6, className='py-2')
            ])
        ], width=10)
    ]),
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col(id='boxscore_datatable', width=10, className='py-2')
    ])
])






##### CALLBACKS (Server/Interactivity) #####
## Display Timeseries Barplot & Averages Line Chart
@callback(
	[Output('timeseries_bar', 'children'),
    Output('probability_hist', 'children')],
	[Input('active_df', 'data'),
	Input('prop_dd', 'value'),
	Input('entity_dd', 'value'),
    Input('entity_type_radio', 'value')]
)
def create_plots(json_df, props, entity, entity_type):
    if props != [] and entity != []:
        df = pd.read_json(json_df, orient="records")
        # # NOTE: FORGET COMBINING PROPS WITH "+" FOR NOW -- WILL ADD THE OPTION TO STAT DROPDOWN LATER (KEEP PROPS INDIVIDUAL!)
        ts_bar = get_bar_ts(df, props, entity)
        ts_l5_avg = get_rolling_line(df, props, 5, entity)
        ts_l10_avg = get_rolling_line(df, props, 10, entity)
        ts_layout = dbc.Col([
            dbc.Row(dcc.Graph(figure=ts_bar)),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=ts_l5_avg), width=6, className='pt-2'),
                dbc.Col(dcc.Graph(figure=ts_l10_avg), width=6, className='pt-2'),
            ])
        ], width=12)
        # Get Probability Plots
        props_hist = get_histogram(df, props, entity)
        under_cdf = get_cdf_line(df, props, False, entity_type) # CDF (UNDER)
        over_cdf = get_cdf_line(df, props, True, entity_type) # CDF (OVER)
        probs_layout = dbc.Col([
            dbc.Row(dcc.Graph(figure=props_hist)),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=under_cdf), width=6, className='pt-2'),
                dbc.Col(dcc.Graph(figure=over_cdf), width=6, className='pt-2'),
            ])
        ], width=12)
        return ts_layout, probs_layout
    return [], []

## Display Entity Name
@callback(
	Output('entity_dashboard_title', 'children'),
	[Input('entity_dd', 'value'),
	Input('active_df', 'data'),
    Input('entity_type_radio', 'value')]
)
def create_entity_title(entity, json_df, entity_type):
    if entity != []:
        df = pd.read_json(json_df, orient="records")
        team_name = get_team_name(df['TEAM'].iloc[-1]) # latest team
        if entity_type == "TEAM":
            return html.H1(f"{team_name} ({entity})")
        else: # PLAYER
            return html.H1(f"{entity} ({team_name})")
    return []

## Display Card Averages for each Entity Prop
@callback(
        Output('cards_section', 'children'),
        [Input('active_df', 'data'),
         Input('prop_dd', 'value')]
)
def create_card_averages(json_df, props):
    if props != [] and json_df != None:
        df = pd.read_json(json_df, orient="records")
        cards = get_cards(df, props) # list of dbc.Card()
        cards_layout = dbc.Row([
            dbc.Col(card, width="auto") for card in cards # cards is a list of dbc.Card()
        ])#, className="px-3 py-1") # "px-3 py-1" controls padding (larger N=3 -> more padding)
        return [cards_layout]
    return []

## Clear Active Parlay DataFrame
@callback(
        [Output('active_parlay_df', 'data', allow_duplicate=True),
         Output('reset_props_list_bttn', 'n_clicks')],
        Input('reset_props_list_bttn', 'n_clicks'),
        prevent_initial_call=True
)
def reset_props_list_df(n_clicks):
    if n_clicks > 0:
        return None, 0

## Display Active Parlay DataTable
@callback(
        Output('parlay_datatable', 'children'),
        Input('active_parlay_df', 'data')
)
def props_list_datatable(json_df):
    if json_df is None: # nothing stored yet
        temp = pd.DataFrame({'Entity': [], 'Prop': [], 'O/U': [], 'Value': [], 'Probability': []})
        dt =  dash_table.DataTable(data=temp.to_dict('records'), columns=[{"name": i, "id": i} for i in temp.columns], style_cell={'textAlign': 'left'})
        return [
            dbc.Row([html.H3("Parlay Probability")]), 
            dbc.Row([dt]),
        ]
    else:
        # Process Dataset
        df = pd.read_json(json_df, orient="records")
        df.columns = ['Entity', 'Prop', 'O/U', 'Value', 'Probability']
        df = df[df['Entity'] != 'TOTAL PROBABILITY'] # AVOIDS RESHUFFLING 'TOTAL PROBABILITY' FROM THE LAST ROW
        df = df.groupby(['Entity','Prop']).last().reset_index() # returns last (latest) row for every group
        df.loc[df.shape[0]] = ['TOTAL PROBABILITY', '', '', '', df['Probability'].product()]
        df['Percentage'] = round(df['Probability'] * 100, 2)
        # Display Props List Table
        dt = dash_table.DataTable( # more at: https://dash.plotly.com/datatable/interactivity
            id='prop_list_datatable',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_data_conditional = [ # color probability column
                {
                    'if': {'column_id': 'Probability', 'row_index': i},
                    'backgroundColor': get_prob_dt_cell_colour(df['Probability'].iloc[i]) # issue before - background colour has to be singular not multiple (if applying to whole or singular column)
                } for i in range(df.shape[0])
            ],
            style_cell={'textAlign': 'left'},
            # row_selectable="multi",
            # selected_rows=[]
        )
        reset_bttn = html.Button('Clear All', id='reset_props_list_bttn', n_clicks=0)
        return [
            dbc.Row([html.H3("Parlay Probability")]), 
            dbc.Row([dt]),
            reset_bttn
        ]

## Store Active Props List Dataframe
@callback(
        [Output('active_parlay_df', 'data'),
         Output('sliders_bttn', 'n_clicks')],
        [Input('active_df', 'data'),
         Input({"type": "prop-slider", "index": ALL}, 'value'),
         Input('sliders_bttn', 'n_clicks'),
         Input('prop_dd', 'value'),
         Input('over_under_radio', 'value'),
         Input('entity_dd', 'value'),
         Input('entity_type_radio', 'value'),
         Input('active_parlay_df', 'data')] # ALL things with the id {'type': 'prop-slider'} are gathered here!!!
)
def store_props_list_df(json_df, slider_values, n_clicks, props, o_u, entity, entity_type, probs_json_df):
    if n_clicks > 0: # Note: button only appears once props and entity inputs are given (so current entity data should be stored already)
        # Get data for plot
        entity_df = pd.read_json(json_df, orient="records").set_index(entity_type)
        counts_df = get_entity_props_counts(entity_df, props, entity_type)
        # Get each slider inputs corresponding probability
        prop_probs = []
        for i, prop in enumerate(props):
            slider_value = slider_values[i]
            prop_prob = get_input_probability(counts_df, prop, slider_value, o_u)
            prop_probs.append( [entity, prop, o_u, slider_value, prop_prob] )
        # Convert to DataFrame
        new_probs_df = pd.DataFrame(prop_probs)
        # Append to existing data and Store as json
        current_probs_df = None # placeholder
        if probs_json_df is None: # first time
            current_probs_df = new_probs_df
        else: # otherwise append to existing
            current_probs_df = pd.read_json(probs_json_df, orient="records")
            current_probs_df = pd.concat([current_probs_df, new_probs_df])
        probs_df_json = current_probs_df.to_json(orient='records')
        return probs_df_json, 0 # only add to current_output if clicked + always reset button state (regardless if pressed or not)

## Display DataTable of Current Entity Props Inputs Probabilitities
@callback(
    Output('odds_table_section', 'children'),
    [Input({"type": "prop-slider", "index": ALL}, 'value'), # ALL things with the id {'type': 'prop-slider'} are gathered here!!!
    Input('active_df', 'data'),
    Input('prop_dd', 'value'),
    Input('entity_dd', 'value'),
    Input('entity_type_radio', 'value'),
    Input('over_under_radio', 'value')]
)
def display_probs_table(slider_values, json_df, props, entity, entity_type, over_under_type):
    if props == [] or entity == [] or slider_values == []:
        return []
    else:
        # Get data for plot
        entity_df = pd.read_json(json_df, orient="records").set_index(entity_type)
        counts_df = get_entity_props_counts(entity_df, props, entity_type)
        # Get each slider inputs corressponding probability
        prop_probs, multi_odds = {}, 1
        for i, prop in enumerate(props):
            slider_value = slider_values[i]
            sign = {"OVER": ">=", "UNDER": "<="}[over_under_type]
            prop_prob = get_input_probability(counts_df, prop, slider_value, over_under_type)
            prop_probs[f'{prop} {sign} {slider_value}'] = prop_prob
            multi_odds *= prop_prob
        prop_probs['Product'] = multi_odds
        # Display Table to represent Additive/Multiplicative Odds
        odds_df = pd.DataFrame.from_dict(prop_probs, orient='index', columns=['Probability']).reset_index(names='Prop')
        odds_dt = [html.Div([
            html.H3("Prop Odds"),
            get_probs_datatable(odds_df) # dash_table.DataTable()
        ])]
        return odds_dt

## Generate dynamic sliders based on prop(s) selected
@callback(
        Output('sliders_section', 'children'),
        [Input('active_df', 'data'),
        Input('prop_dd', 'value'),
        Input('entity_dd', 'value')]
)
def display_prop_sliders(json_df, props, entity):
    if props != [] and entity != []:
        df = pd.read_json(json_df, orient="records")
        ou_radio = dbc.Row([
            dbc.Col(html.Label("OVER or UNDER:"), width="auto"),
            dbc.Col(
                dcc.RadioItems(
                    id="over_under_radio",
                    options=["OVER", "UNDER"],
                    value="OVER",
                    inline=True,
                    labelStyle={'margin-right': '15px'}
                ), width="auto")
        ])
        sliders_section = get_sliders_section(df, props) # list of divs which contain sliders
        bttn = html.Button('Add To Parlay', id='sliders_bttn', n_clicks=0)
        return dbc.Col([
            ou_radio,
            dbc.Row(sliders_section),
            bttn
        ])
    return []

## Display active dataframe
@callback(
        Output('boxscore_datatable', 'children'),
        [Input('active_df', 'data'),
         Input('prop_dd', 'value')]
)
def display_active_df(json_df, props):
    if json_df != None and props != []:
        df = pd.read_json(json_df, orient="records").iloc[::-1].reset_index(drop=True) # latest games first
        if len(props) > 1:
            combined_prop_name  = "+".join(props)
            df[combined_prop_name] = df[props].sum(axis=1) # row sums
            props.append(combined_prop_name) # add combined prop as prop
        # Get Cell Colours for Prop Columns (Needs to be standardised to WHOLE column - not just rows shown)
        colors = {}
        for prop in props:
            col_colours = [get_cell_colour(z_score) for z_score in get_z_score(df[prop])] # calculate z_scores for column then get colour
            colors[prop] = col_colours
        # DataTable
        dt = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_data_conditional = [ # color probability column
                {
                    'if': {'column_id': prop, 'row_index': i},
                    'backgroundColor': colors[prop][i]
                } for prop in props for i in df.index
            ],
            style_cell={'textAlign': 'left'},
            filter_action="native"
        )
        return [html.H4("Previous Games"), dt]

## Store the active filtered/subset dataframe
@callback(
        Output('active_df', 'data'),
        [Input('entity_type_radio', 'value'),
         Input('entity_dd', 'value'),
         Input('prop_dd', 'value')]
)
def store_active_df(team_player_choice, entity, stat_choice):
    temp_df = None # Select correct dataset and filter/subset
    if team_player_choice == "PLAYER":
        temp_df = players[["TEAM", "GAME DATE", "TEAM OPP", "HOME"] + stat_choice] # subset default columns
    else:
        temp_df = teams[["GAME DATE", "TEAM OPP", "HOME"] + stat_choice]
    temp_df = temp_df.loc[entity] # filter by chosen entities
    if len(stat_choice) > 1:
        prop_name = "+".join(stat_choice)
        temp_df[prop_name] = temp_df[stat_choice].sum(axis=1) # attach row sums
    return temp_df.reset_index().to_json(orient="records") # re-add index as column

## Update entity dropdown options and statistics dropdown
@callback(
    [Output('entity_dd', 'options'),
     Output('entity_dd', 'value'),
     Output('prop_dd', 'options'),
     Output('prop_dd', 'value')],
    Input('entity_type_radio', 'value')
)
def update_entity_dd_options(team_player_choice):
    if team_player_choice == "PLAYER":
        return player_dd_opts, [], ['PTS','REB','AST','3PM','BLK','STL'], []
    else:
        return team_dd_opts, [], ['PTS','+/-'], []
    
if __name__ == '__main__':
    app.run(debug=False) # hot-reloading - Dash will automatically refresh browser when changes in code
