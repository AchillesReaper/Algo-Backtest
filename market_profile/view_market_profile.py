import ast
from typing import Dict
import pandas as pd
from dash import Dash, html, dcc, Output, Input, State, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from typing import Dict
import pandas as pd
from dash import Dash, html, dcc, Output, Input, State, dash_table
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from termcolor import cprint
from dh_backtest.models.data_classes import Underlying, IBBarSize
# local modules
from get_market_profile import gen_market_profile, get_data_for_mp
from view_css import style_root_div, style_header, style_body, style_body_sub_div, style_element



def plot_market_profile(tpo_dic:Dict, df_td_trade:pd.DataFrame) -> go.Figure:
    '''
    This is custom function to plot the candlestick chart for the trade date.
    '''
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.6],
        shared_yaxes=True,
    )
    fig.add_trace(
        go.Bar(
            x=list(tpo_dic.values()),
            y=list(tpo_dic.keys()),
            name='TPO',
            orientation='h',
            marker_color='blue',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=df_td_trade.index,
            open=df_td_trade['open'],
            high=df_td_trade['high'],
            low=df_td_trade['low'],
            close=df_td_trade['close'],
            name='candlestick',
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#F8EDE3',
        xaxis2=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    return fig


def plot_mp_app(underlying:Underlying, is_read_data:bool=False) -> go.Figure:
    '''
    This is custom function to visualize the market profile.
    '''
    if is_read_data:
        df_clean    = pd.read_csv('data/market_profile/hsi_day_1_clean.csv', index_col=0)
        df_mp       = pd.read_csv('data/market_profile/hsi_day_1_mp.csv', index_col=0)
    else:
        df_clean    = get_data_for_mp(underlying)
        df_mp       = gen_market_profile(df_clean)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7],
    )
    fig.update_layout(
        height=None,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#F8EDE3',
    )
    app = Dash()
    app.layout = html.Div(
        style    = style_root_div,
        children = [
            html.Div(
                id          ="header", 
                style       =style_header,
                className   ='row', 
                children    ='Market Profile',
            ),
            html.Div(
                id      ='body',
                style   =style_body,
                children=[
                    dcc.Store(id='current_ref', data=''),
                    html.Div(
                        style   = {**style_body_sub_div, 'width': '60%'},
                        children= [
                            dcc.Graph(id='graph_area', figure=fig, style=style_element),
                        ]
                    ),
                    html.Div(
                        style={**style_body_sub_div, 'width': '35%'},
                        children = [
                            html.Div(
                                style=style_element,
                                children = [
                                    dash_table.DataTable(
                                        id='price_table',
                                        data=df_mp[['close', 'skewness', 'kurtosis']].copy().reset_index().to_dict('records'),
                                        columns=[
                                            {'name': 'Trade Date', 'id': 'trade_date'},
                                            {'name': 'Close', 'id': 'close', 'type': 'numeric'},
                                            {'name': 'Skewness', 'id': 'skewness', 'type': 'numeric'},
                                            {'name': 'Kurtosis', 'id': 'kurtosis', 'type': 'numeric'},
                                        ],
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'Close'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'Skewness'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'Kurtosis'}, 'textAlign': 'right'},
                                        ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                        page_size=20,
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            ),
            html.Div(
                id='footer',
                style={'width': '100%'},
                children=[]
            )
        ]
    )

    @app.callback(
        [
            # Output('current_ref', 'data'),
            Output('price_table', 'style_data_conditional'),
            Output('graph_area', 'figure'),
        ],
        Input('price_table', 'active_cell'),
        [
            State('price_table', 'data'),
        ],
    )
    def update_fig(active_cell, tableData):
        if active_cell is None:
            raise PreventUpdate
        
        # retrieve the trade date from the active cell
        td = tableData[active_cell['row']]['trade_date']
        print(f'active date: {td}')
        # highlight the selected row
        style_data_conditional = [{
            'if': {'filter_query': f'{{trade_date}} eq "{td}"'},
            'backgroundColor': 'lightblue'
        }]
        # retrieve the data for the selected trade date
        df_td_trade = df_clean[df_clean['trade_date'] == td]
        tpo_dic     = df_mp[df_mp.index == td]['tpo_count'].to_dict()[td]
        tpo_dic     = ast.literal_eval(tpo_dic)
        # print(list(tpo_dic.keys()))
        fig         = plot_market_profile(tpo_dic, df_td_trade)

        return style_data_conditional, fig
    
    app.run(debug=True)

if __name__ == "__main__":
    underlying = Underlying(
        symbol='HSI',
        exchange='HKFE',
        contract_type='FUT',
        barSizeSetting=IBBarSize.DAY_1,
        start_date='2024-01-01',
        end_date='2024-01-31',
    )
    # folder_path='data/market_profile'
    # if not os.path.exists(folder_path): os.makedirs(folder_path)
    # df_clean    = get_data_for_mp(underlying)
    # df_mp       = gen_market_profile(df_clean)
    # df_clean.to_csv(f'{folder_path}/hsi_day_1_clean.csv', index=True)
    # df_mp.to_csv(f'{folder_path}/hsi_day_1_mp.csv', index=True)
    plot_mp_app(underlying, True)
