from typing import List
import pandas as pd
import dash
from dash import Dash, html, dcc, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from termcolor import cprint
# local modules
from view_css import style_root_div, style_header, style_body, style_body_sub_div, style_element
# from dh_backtest.views.css import style_root_div, style_header, style_body, style_body_sub_div, style_element


df_bt_result_list = []
df_performance  = ''
df_para         = ''


def plot_detail(df_bt_result:pd.DataFrame):
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    # price movement of the underlying
    fig.add_trace(
        go.Candlestick(
            x       =df_bt_result['datetime'],
            open    =df_bt_result['open'],
            high    =df_bt_result['high'],
            low     =df_bt_result['low'],
            close   =df_bt_result['close'],
            name    ='Price',
        ),
        secondary_y=False
        )
    # equity curve
    fig.add_trace(
        go.Scatter(
            x       =df_bt_result['datetime'],
            y       =df_bt_result['nav'],
            mode    ='lines',
            name    ='Equity',
            line    =dict(color='green', width=2),
            customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
        ),
        secondary_y=True,
    )
    # actions - buy
    action_buy_df = df_bt_result[df_bt_result['action'] == 'buy']
    fig.add_trace(
        go.Scatter(
            x       =action_buy_df['datetime'],
            y       =action_buy_df['t_price'],
            mode    ='markers',
            marker  =dict(
                symbol  ='triangle-up-open',
                size    =10,
                color   ='brown',
            ),
            name='Buy',
            text='Open: ' 
                + action_buy_df['t_size'].astype(str) + '@' + action_buy_df['t_price'].astype(str) 
                + ' (signal: ' + action_buy_df['signal'] + ')',
            hoverinfo='text',
            customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
        ),
        secondary_y=False,
    )
    # actions - sell
    action_sell_df = df_bt_result[df_bt_result['action'] == 'sell']
    fig.add_trace(
        go.Scatter(
            x       =action_sell_df['datetime'],
            y       =action_sell_df['t_price'],
            mode    ='markers',
            marker  =dict(
                symbol='triangle-down-open',
                size=10,
                color='brown',
            ),
            name='Sell',
            text='Open: ' 
                + action_sell_df['t_size'].astype(str) + '@' + action_sell_df['t_price'].astype(str) 
                + ' (signal: ' + action_sell_df['signal'] + ')',
            hoverinfo='text',
            customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
        ),
        secondary_y=False,
    )
    #actions - close
    action_close_df = df_bt_result[df_bt_result['action'] == 'close']
    fig.add_trace(
        go.Scatter(
            x=action_close_df['datetime'],
            y=action_close_df['t_price'],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=10,
                color='blue',
            ),
            name='Close',
            text='Close: ' 
                + action_close_df['t_size'].astype(str) + '@' + action_close_df['t_price'].astype(str) 
                + ' (' + action_close_df['logic'] + ', P/L: ' + action_close_df['pnl_action'].astype(str) + ')',
            hoverinfo='text',
            customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        yaxis=dict(tickformat=',', autorange=True),
        yaxis2=dict(tickformat=',', autorange=True),
        xaxis=dict(showticklabels=True, autorange=True, type='date'),
        autosize=True,
        xaxis_rangeslider_visible=False,
        height=800,
        hovermode='closest',
        hoverlabel=dict(bgcolor='#af9b46', font_size=16, font_family='Rockwell',),
        paper_bgcolor='#F8EDE3',
    )

    return fig

def plot_all(df_bt_result_list: List[pd.DataFrame]):
    fig = go.Figure()
    for df in df_bt_result_list:
        fig.add_trace(go.Scatter(
            x       = df['datetime'], 
            y       = df['nav'], 
            mode    = 'lines', 
            name    = 'nav',
            line    = {'width': 2},
            customdata = [df.attrs['ref_tag']] * len(df),
            text    =   f'Ref: {df.attrs["ref_tag"]} <br>' +
                        f'total_trades: {df.attrs["performace_report"]["number_of_trades"]} <br>' +
                        f'win_rate: {df.attrs["performace_report"]["win_rate"]:.2f} <br>' +
                        f'total_cost: {df.attrs["performace_report"]["total_cost"]:,.2f} <br>' +
                        f'pnl $: {df.attrs["performace_report"]["pnl_trading"]:,.2f} <br>' +
                        f'roi %: {df.attrs["performace_report"]["roi_trading"]:.2%} <br>' +
                        f'mdd $: {df.attrs["performace_report"]["mdd_dollar_trading"]:,.2f} <br>' +
                        f'mdd %: {df.attrs["performace_report"]["mdd_pct_trading"]:.2%} <br>' +
                        f'roi(trading-B&H) %: {(df.attrs["performace_report"]["roi_trading"]-df.attrs["performace_report"]["roi_bah"]):.2%} <br>' +
                        f'mdd(trading-B&H) %: {(df.attrs["performace_report"]["mdd_pct_trading"]-df.attrs["performace_report"]["mdd_pct_bah"]):.2%} <br>',
            hoverinfo='text',
        ))

    fig.update_layout(
        height=None,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#F8EDE3',
    )

    return fig

def plot_app(df_list: List[pd.DataFrame]):
    global df_bt_result_list
    df_bt_result_list = df_list

    global df_performance
    df_performance = pd.DataFrame(columns=['ref_tag', 'number_of_trades', 'win_rate', 'total_cost', 'pnl_trading', 'roi_trading', 'mdd_pct_trading', 'mdd_dollar_trading', 'pnl_bah', 'roi_bah', 'mdd_pct_bah', 'mdd_dollar_bah'])
    
    global df_para
    df_para_columns = ['ref_tag'] + (list(df_bt_result_list[0].attrs['para_comb'].keys()))
    df_para = pd.DataFrame(columns=df_para_columns)

    fig = plot_all(df_bt_result_list)

    for df in df_bt_result_list:
        df_performance.loc[df.attrs['ref_tag']] = [
            df.attrs['ref_tag'],
            df.attrs['performace_report']['number_of_trades'],
            df.attrs['performace_report']['win_rate'],
            df.attrs['performace_report']['total_cost'],
            df.attrs['performace_report']['pnl_trading'],
            df.attrs['performace_report']['roi_trading'],
            df.attrs['performace_report']['mdd_pct_trading'],
            df.attrs['performace_report']['mdd_dollar_trading'],
            df.attrs['performace_report']['pnl_bah'],
            df.attrs['performace_report']['roi_bah'],
            df.attrs['performace_report']['mdd_pct_bah'],
            df.attrs['performace_report']['mdd_dollar_bah']
        ]

        df_para.loc[df.attrs['ref_tag']] = [df.attrs['ref_tag']] + list(df.attrs['para_comb'].values())

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    money       = dash_table.FormatTemplate.money(2)
    percentage  = dash_table.FormatTemplate.percentage(2)

    app.layout = html.Div(
        style    = style_root_div,
        children = [
            html.Div(
                id          ="header", 
                style       =style_header,
                className   ='row', 
                children    ='Backtest Result',
            ),
            html.Div(
                id      ='body',
                style   =style_body,
                children=[
                    dcc.Store(id='current_ref', data=''),
                    html.Div(
                        style   = {**style_body_sub_div, 'width': '60%'},
                        children= [
                            dbc.Tabs(
                                [
                                    dbc.Tab(label='All Curves', tab_id='all_curves'),
                                    dbc.Tab(label='Strategy Detail',tab_id='strategy_detail'),
                                ],
                                id='graph_tab',
                                active_tab='all_curves',
                            ),
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
                                        id='bt_result_table',
                                        data=df_performance[['ref_tag', 'pnl_trading', 'roi_trading', 'mdd_pct_trading']].to_dict('records'),
                                        columns=[
                                            {'name': 'Backtest Reference', 'id': 'ref_tag'},
                                            {'name': 'Profit/Loss', 'id': 'pnl_trading', 'type': 'numeric', 'format': money},
                                            {'name': 'ROI', 'id': 'roi_trading', 'type': 'numeric', 'format': percentage},
                                            {'name': 'MDD', 'id': 'mdd_pct_trading', 'type': 'numeric', 'format': percentage},
                                        ],
                                        sort_by=[{'column_id': 'roi_trading', 'direction': 'desc'}],
                                        sort_action='native',
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'pnl_trading'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'roi_trading'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'mdd_pct_trading'}, 'textAlign': 'right'},

                                        ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                        page_size=8,
                                    )
                                ]
                            ),
                            html.Div(
                                id='performance_table',
                                style=style_element,
                                children=[]
                            ),
                            html.Div(
                                id='para_table',
                                style=style_element,
                                children=[]
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
        Output('bt_result_table', 'data'),
        Input('bt_result_table', 'sort_by'),
        State('bt_result_table', 'data')
    )
    def update_table_data(sort_by, tableData):
        if not sort_by:
            raise PreventUpdate

        df = pd.DataFrame(tableData)
        for sort in sort_by:
            df = df.sort_values(by=sort['column_id'], ascending=(sort['direction'] == 'asc'))

        return df.to_dict('records')

    # update state of current reference
    @app.callback(
        Output('current_ref', 'data'),
        [Input('graph_area', 'clickData'), Input('bt_result_table', 'active_cell'),],
        State('bt_result_table', 'data')
    )
    def update_current_ref(clickData, active_cell, tableData):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        cprint(f'trigger_id: {trigger_id} -> upadte current ref', 'yellow')

        if trigger_id == 'graph_area' and clickData:
            ref_tag = clickData['points'][0]['customdata']
            return ref_tag
        
        if trigger_id == 'bt_result_table' and active_cell:
            ref_tag = tableData[active_cell['row']]['ref_tag']
            return ref_tag
    
    
    # consquence of updating the current reference state
    @app.callback(
        [
            Output('graph_area', 'figure'),
            Output('bt_result_table', 'style_data_conditional'),
            Output('performance_table', 'children'),
            Output('para_table', 'children'),
        ],
        [
            Input('current_ref', 'data'),
            Input('graph_tab', 'active_tab'),
        ],
        [
            State('graph_area', 'figure'),
            State('bt_result_table', 'style_data_conditional'),
            State('performance_table', 'children'),
            State('para_table', 'children'),
        ],
        allow_duplicate=True
    )
    def update_for_ceuurent_ref(current_ref, active_tab, figure, style_data_conditional, perform_tables, para_table):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        cprint(f'trigger_id: {trigger_id}', 'green')
        
        global df_bt_result_list
        if active_tab == 'strategy_detail':
            if not current_ref:
                raise PreventUpdate
            for df in df_bt_result_list:
                if df.attrs['ref_tag'] == current_ref:
                    figure = plot_detail(df)
        else:
            if active_tab == 'all_curves':
                figure = plot_all(df_bt_result_list)
            if current_ref:
                for trace in figure['data']:
                    if trace['customdata'][0] == current_ref:
                        trace['line']['width'] = 5
                        trace['opacity'] = 1
                    else:
                        trace['line']['width'] = 2
                        trace['opacity'] = 0.7

        style_data_conditional = [{
            'if': {'filter_query': f'{{ref_tag}} eq "{current_ref}"'},
            'backgroundColor': 'lightblue'
        }]

        global df_performance
        df_table1 = pd.DataFrame(
            {
                'Reference':['Number of Trades', 'Win Rate', 'Total Cost', 'PnL Trading', 'ROI Trading'],
                current_ref:[
                    f'{df_performance.loc[current_ref]["number_of_trades"]:,}',
                    f'{df_performance.loc[current_ref]["win_rate"]:.2%}',
                    f'{df_performance.loc[current_ref]["total_cost"]:,.2f}',
                    f'{df_performance.loc[current_ref]["pnl_trading"]:,.2f}',
                    f'{df_performance.loc[current_ref]["roi_trading"]:.2%}',
                ]
            },
        )
        df_table2 = pd.DataFrame(
            {
                'Metrics':['Profit/Loss', 'Return on Investment', 'MDD Dollar', 'MDD Percentage' ],
                'Trading':[
                    f'{df_performance.loc[current_ref]["pnl_trading"]:,.2f}',
                    f'{df_performance.loc[current_ref]["roi_trading"]:.2%}',
                    f'{df_performance.loc[current_ref]["mdd_dollar_trading"]:,.2f}',
                    f'{df_performance.loc[current_ref]["mdd_pct_trading"]:.2%}',
                ],
                'Buy & Hold':[
                    f'{df_performance.loc[current_ref]["pnl_bah"]:,.2f}',
                    f'{df_performance.loc[current_ref]["roi_bah"]:.2%}',
                    f'{df_performance.loc[current_ref]["mdd_dollar_bah"]:,.2f}',
                    f'{df_performance.loc[current_ref]["mdd_pct_bah"]:.2%}',
                ]
            },
        )
        table1 = dash_table.DataTable(
            data=df_table1.to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {'if': {'column_id': current_ref}, 'textAlign': 'right'},
            ],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )
        table2 = dash_table.DataTable(
            data=df_table2.to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {'if': {'column_id': 'Metrics'}, 'fontWeight': 'bold', 'textAlign': 'left'},
                {'if': {'column_id': 'Trading'}, 'backgroundColor': 'lightblue', 'textAlign': 'right'},
                {'if': {'column_id': 'Buy & Hold'}, 'backgroundColor': 'lightgreen', 'textAlign': 'right'}
            ],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )
        
        global df_para
        df_table = pd.DataFrame(
            {
                'para_name': df_para.columns[1:],
                'para_value': df_para.loc[current_ref][1:]
            }
        )
        para_table = dash_table.DataTable(
            data=df_table.to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {'if': {'column_id': 'para_name'}, 'textAlign': 'left', 'fontWeight': 'bold'},
                {'if': {'column_id': 'para_value'}, 'textAlign': 'right'},
            ],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )

        return figure, style_data_conditional, [table1, table2], para_table

    app.run(debug=True)





