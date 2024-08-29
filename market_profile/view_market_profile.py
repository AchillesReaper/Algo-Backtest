import ast
import os
from typing import Dict, List
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



def plot_market_profile(df_tpo:pd.DataFrame, df_td_trade:pd.DataFrame) -> go.Figure:
    '''
    This is custom function to plot the candlestick chart for the trade date.
    '''
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.6],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Bar(
            x = df_tpo['count'],
            y = df_tpo['price'],
            name='TPO',
            orientation='h',
            marker_color=df_tpo['color'],
            text= 'Price:' + df_tpo['price'].astype(str) + ', ' + 'TPO: ' + df_tpo['count'].astype(str),
            textposition='none',
            hoverinfo='text',
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
        xaxis=dict(
            tickmode='linear',
            nticks=len(df_tpo['count']),
            showspikes=True, spikemode='across', spikesnap='cursor', showline=True
        ),
        xaxis2=dict(
            rangeslider=dict(visible=False),
            showspikes=True, spikemode='across', spikesnap='cursor', showline=True
        ),
        yaxis=dict(
            side='right',
            tickformat=',',
            showgrid=True,
            showspikes=True, spikemode='across', spikesnap='cursor', showline=True
        ),
        yaxis2=dict(
            side='right',
            tickformat=',',
            showspikes=True, spikemode='across', spikesnap='cursor', showline=True
        ),
    )
    return fig


def plot_mp_app(underlying:Underlying, is_update_data:bool=False, folder_path:str='') -> go.Figure:
    '''
    This is custom function to visualize the market profile.
    1. if is_read_data is True, read the data from the local file.
    2. if is_read_data is False, get the data from the IB server. <b> resolution </b> comes into play for this case.
    '''
    clean_path  = f'{folder_path}/{underlying.symbol}_clean_{underlying.start_date}_{underlying.end_date}.csv'.replace('-','')
    if is_update_data:
        df_mp       = gen_market_profile(folder_path, underlying, is_update_data=is_update_data)
        df_clean    = pd.read_csv(clean_path, index_col=0)
    else:
        df_clean    = pd.read_csv(clean_path, index_col=0)
        mp_path = f'{folder_path}/{underlying.symbol}_mp_{underlying.start_date}_{underlying.end_date}.csv'.replace('-','')
        df_mp               = pd.read_csv(mp_path, index_col=0)
        df_mp['tpo_count']  = df_mp['tpo_count'].apply(ast.literal_eval)
        df_mp['pocs']       = df_mp['pocs'].apply(ast.literal_eval)


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
                children    =html.H2('Market Profile'),
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
                                        id          ='price_table',
                                        data        =df_mp[['close', 'skewness', 'kurtosis']].copy().reset_index().to_dict('records'),
                                        columns     =[
                                                        {'name': 'Trade Date', 'id': 'trade_date'},
                                                        {'name': 'Close', 'id': 'close', 'type': 'numeric'},
                                                        {'name': 'Skewness', 'id': 'skewness', 'type': 'numeric'},
                                                        {'name': 'Kurtosis', 'id': 'kurtosis', 'type': 'numeric'},
                                                    ],
                                        sort_by     =[{'column_id': 'kurtosis', 'direction': 'asc'}],
                                        sort_action ='native',
                                        active_cell ={'row': 0, 'column': 0, 'column_id': 'trade_date'},
                                        style_cell  ={'textAlign': 'left'},
                                        style_cell_conditional=[
                                                        {'if': {'column_id': 'close'}, 'textAlign': 'right'},
                                                        {'if': {'column_id': 'skewness'}, 'textAlign': 'right'},
                                                        {'if': {'column_id': 'kurtosis'}, 'textAlign': 'right'},
                                                    ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                        page_size=26,
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
        Output('price_table', 'data'),
        Input('price_table', 'sort_by'),
        State('price_table', 'data'),
    )
    def update_table_order(sort_by, tableData):
        if not sort_by:
            raise PreventUpdate
        # cprint(f'tableData type: {type(tableData)}', 'yellow')
        # cprint(f'tableData: {tableData}', 'blue') 
        return sorted(tableData, key=lambda x: x[sort_by[0]['column_id']], reverse=sort_by[0]['direction'] == 'desc')


    def color_tag(price:int, val:int, vah:int, spkl, spkh, pocs:List[int]) -> str:
        '''
        This is custom function to color the price based on the market profile.
        '''
        if price in pocs:
            return 'gold'
        elif (price >= val and price <= vah):
            return 'green'
        elif (price <= spkl or price >= spkh):
            return 'yellow' 
        else:
            return 'blue'

    # show figure for the selected trade date
    @app.callback(
        [
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
        tpo_dict    = df_mp[df_mp.index == td]['tpo_count'].to_dict()[td]
        pocs        = list(map(int, df_mp[df_mp.index == td]['pocs'].to_dict()[td]))
        val         = df_mp[df_mp.index == td]['val'].to_dict()[td]
        vah         = df_mp[df_mp.index == td]['vah'].to_dict()[td]
        spkl        = df_mp[df_mp.index == td]['spkl'].to_dict()[td]
        spkh        = df_mp[df_mp.index == td]['spkh'].to_dict()[td]


        df_tpo = pd.DataFrame(tpo_dict.items(), columns=['price', 'count'])

        df_tpo['color'] = [color_tag(price, val, vah, spkl, spkh, pocs) for price in df_tpo['price']]
        fig = plot_market_profile(df_tpo, df_td_trade)

        return style_data_conditional, fig
    
    app.run(debug=True)

if __name__ == "__main__":
    underlying = Underlying(
        symbol='HSI',
        exchange='HKFE',
        contract_type='FUT',
        barSizeSetting=IBBarSize.DAY_1,
        start_date='2024-01-01',
        end_date='2024-03-31',
    )
    folder_path='data/market_profile'
    is_update_data = False

    plot_mp_app(underlying, is_update_data, folder_path=folder_path)
