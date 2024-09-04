'''
testing mean reversion strategy
step 1: define spikes
step 2: if market open on spike -> forecasting price reverse to the nearest POC
'''

import ast
from datetime import datetime
import json
import sys
import numpy as np
import pandas as pd
from termcolor import cprint
from dh_backtest.models.data_classes import Underlying, IBBarSize, FutureTradingAccount
from dh_backtest.models.remote_data import get_spot_future_ib
from dh_backtest.backtest_engine import BacktestEngine
# local imports
from get_market_profile import gen_market_profile
from visual_bt_results import plot_app


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)


def generate_signal(df:pd.DataFrame, para_comb:dict, underlying:Underlying) -> pd.DataFrame:
    '''
    this is custom function to generate signals based on the historical data.
    return the input dataframe with extra column ['calculation_col_1', 'calculation_col_2', 'signal'].
    '''
    # trim df and limit time before 10:45am
    df.reset_index(drop=False, inplace=True)
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.between_time('8:00', '10:45')
    df.reset_index(drop=False, inplace=True)

    is_update_data = False
    if is_update_data:
        df_mp = gen_market_profile('data/market_profile', underlying, is_update_data=True)
    else:
        df_mp = pd.read_csv(f'data/market_profile/{underlying.symbol}_mp_{underlying.start_date}_{underlying.end_date}.csv'.replace('-',''), index_col=0)
        df_mp['tpo_count']  = df_mp['tpo_count'].apply(ast.literal_eval)
        df_mp['pocs']       = df_mp['pocs'].apply(ast.literal_eval)
    
    start_date_stamp = datetime.strptime(underlying.start_date, '%Y-%m-%d').timestamp() + 3600*8
    df_mp['for_td'] = df_mp.index.to_series().shift(-1)
    df = pd.merge(df, df_mp[['for_td', 'pocs', 'val', 'vah', 'spkl', 'spkh', 'tpo_count']], left_on='trade_date', right_on='for_td', how='left')
    df = df[df['timestamp'] > start_date_stamp]
    df.set_index('timestamp', inplace=True)

    df['signal'] = np.where(
        df['trade_date'] != df['trade_date'].shift(1),
          np.where(
            (df['close'] < df['spkl']) & (df['close'] > df['tpo_count'].apply(lambda x: list(x.keys())[0])),
            'buy',
            np.where(
                (df['close'] > df['spkh']) & (df['close'] < df['tpo_count'].apply(lambda x: list(x.keys())[0])),
                'sell',
                'no'
            )
          ),
          ''
    )
    return df


def action_on_signal(df, para_comb, trade_account) -> pd.DataFrame:
    '''
    this is custom function for traders to determine what to do with their custome signal.
    return the input dataframe with extra column ['action', 'logic', 't_price', 't_size', 'commission', 'pnl_action' 'acc_columns'].
    the action can be 'buy', 'sell', and 'close' only.
    logic is the reason for the action. suggestion:['buy on signal', 'sell on signal', 'close on signal', 'stop loss limit', 'target profit', 'force close', 'margin call']
    t_price is the price to execute the action.
    t_size is the size of the position to execute the action.
    commission is the commission to be charged for the action.
    pnl_action is the realized profit and loss due to the action.
    acc_columns is the columns recording the changes of the trading account.
    '''
    my_acc = trade_account
    my_acc.oco = {'amount': 0}
    stop_torlorance = 1 + float(para_comb['stop_loss'].split(':')[1])


    # duplicate the row after the row with signal -> for the case open and close position in the same bar
    target_row_indices = df.index[df['signal'].shift(1).isin(['buy', 'sell'])].tolist()
    rows_to_duplicate = df.loc[target_row_indices]
    rows_to_duplicate.index = rows_to_duplicate.index + 60
    rows_to_duplicate['datetime'] = rows_to_duplicate['datetime'] + pd.Timedelta(minutes=1)
    
    df = pd.concat([df, rows_to_duplicate], axis=0)
    df = df.sort_index()

    is_signal_buy = False
    is_signal_sell= False
    for index, row in df.iterrows():
        ''' Strategy: 
        1. if signal is buy and current position long or zero, add a long position
        2. if signal is sell and current position short or zero, add a short position
        3. if the signal inicate different direction from current position, skep this step
        '''
        initial_margin_per_contract = row['open']* my_acc.contract_multiplier * my_acc.margin_rate
        if is_signal_buy or is_signal_sell:
            # step 1: determine if it is time to open position
            if my_acc.bal_avialable > initial_margin_per_contract:
                if is_signal_buy and my_acc.position_size >= 0:
                    commission          = my_acc.open_position(1, row['open'])
                    df.loc[index, 'action']       = 'buy'
                    df.loc[index, 'logic']        = 'open'
                    df.loc[index, 't_size']       = 1
                    df.loc[index, 't_price']      = row['open']
                    df.loc[index, 'commission']   = commission
                    df.loc[index, 'pnl_action']   = -commission
                    stop_loss_level = list(row['tpo_count'].keys())[0] * stop_torlorance
                    if para_comb['target_profit'] == 'first_poc':
                        my_acc.oco  = {'amount': -1, 'target': row['pocs'][0], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'last_poc':
                        my_acc.oco  = {'amount': -1, 'target': row['pocs'][-1], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'close_va_b':
                        my_acc.oco  = {'amount': -1, 'target': row['val'], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'far_va_b':
                        my_acc.oco  = {'amount': -1, 'target': row['vah'], 'stop': stop_loss_level}
                    else:
                        pass
                    df.loc[index, 'oco'] = json.dumps(my_acc.oco)

                elif is_signal_sell and my_acc.position_size <= 0:
                    commission          = my_acc.open_position(-1, row['open'])
                    df.loc[index, 'action']       = 'sell'
                    df.loc[index, 'logic']        = 'open'
                    df.loc[index, 't_size']       = -1
                    df.loc[index, 't_price']      = row['open']
                    df.loc[index, 'commission']   = commission
                    df.loc[index, 'pnl_action']   = -commission
                    stop_loss_level = list(row['tpo_count'].keys())[-1] * stop_torlorance
                    if para_comb['target_profit'] == 'first_poc':
                        my_acc.oco  = {'amount': 1, 'target': row['pocs'][-1], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'last_poc':
                        my_acc.oco  = {'amount': 1, 'target': row['pocs'][0], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'close_va_b':
                        my_acc.oco  = {'amount': 1, 'target': row['vah'], 'stop': stop_loss_level}
                    elif para_comb['target_profit'] == 'far_va_b':
                        my_acc.oco  = {'amount': 1, 'target': row['val'], 'stop': stop_loss_level}
                    else:
                        pass
                    df.loc[index, 'oco'] = json.dumps(my_acc.oco)
            else:
                pass
        else:
            # step 2: determine if it is time to close position        
            is_close_position = False
            if my_acc.oco['amount'] != 0:
                if (df.loc[index, 'datetime'].hour >= 10) & (df.loc[index, 'datetime'].fromtimestamp(index).minute>=30):
                    # reached the max holding period
                    is_close_position = True
                    df.loc[index, 'logic']    = 'max holding period'
                    df.loc[index, 't_price']  = row['open']
                elif my_acc.oco['target'] in range(row['low'], row['high']+1):
                    # reached target price
                    is_close_position = True
                    df.loc[index, 'logic']    = 'reach profit target'
                    df.loc[index, 't_price']  = my_acc.oco['target']
                elif my_acc.oco['stop'] in range(row['low'], row['high']+1):
                    # reached stop loss
                    is_close_position = True
                    df.loc[index, 'logic']    = 'stop loss'
                    df.loc[index, 't_price']  = my_acc.oco['stop']
            if is_close_position:
                df.loc[index, 'action']                       = 'close'
                df.loc[index, 't_size']                       = my_acc.oco['amount']
                df.loc[index, 'commission'], df.loc[index, 'pnl_action']= my_acc.close_position(my_acc.oco['amount'], df.loc[index, 't_price'])
                my_acc.oco = {'amount':0}        

        # step 3: update the account and record the action if any
        mtm_result =  my_acc.mark_to_market(row['close'])

        if mtm_result['signal'] == 'margin call':
            df.loc[index, 'action'] = mtm_result['action']
            if mtm_result['action'] == 'close':
                df.loc[index, 'logic'] = mtm_result['logic']
                df.loc[index, 'commission'] = mtm_result['commission']
                df.loc[index, 'pnl_action'] = mtm_result['pnl_realized']

        df.loc[index, 'pos_size']       = int(my_acc.position_size)
        df.loc[index, 'pos_price']      = float(my_acc.position_price)
        df.loc[index, 'pnl_unrealized'] = float(my_acc.pnl_unrealized)
        df.loc[index, 'nav']            = float(my_acc.bal_equity)
        df.loc[index, 'bal_cash']       = float(my_acc.bal_cash)
        df.loc[index, 'bal_avialable']  = float(my_acc.bal_avialable)
        df.loc[index, 'margin_initial'] = float(my_acc.margin_initial)
        df.loc[index, 'cap_usage']      = f'{my_acc.cap_usage:.2f}%'

        # update the signal status
        match para_comb['open_direction']:
            case 'long_only':
                is_signal_sell= False
                is_signal_buy = True if (row['signal'] == 'buy') and (row['volume'] > 0) else False   
            case 'short_only':
                is_signal_buy = False
                is_signal_sell= True if (row['signal'] == 'sell') and (row['volume'] > 0) else False
            case 'whatever':
                is_signal_buy = True if row['signal'] == 'buy' else False
                is_signal_sell= True if row['signal'] == 'sell' else False

    return df



if __name__ == "__main__":
    underlying = Underlying(
        symbol='HSI',
        exchange='HKFE',
        contract_type='FUT',
        barSizeSetting=IBBarSize.MIN_5,
        start_date='2024-01-01',
        end_date='2024-08-30',
    )


    para_dict = {
        'open_direction'    : ['whatever'],
        'stop_loss'         : ['spike:0.0', 'spike:0.01'],
        'target_profit'     : ['first_poc'],
    }

    engine = BacktestEngine(
        is_update_data      = False,
        is_rerun_backtest   = False,
        # is_rerun_backtest   = False,
        underlying          = underlying,
        para_dict           = para_dict,
        trade_account       = FutureTradingAccount(150_000),
        generate_signal     = generate_signal,
        action_on_signal    = action_on_signal,
        get_data_from_api   = get_spot_future_ib,
        folder_path         = 'data/market_profile',
        plot_app            = plot_app,
    )

    engine.run_engine()


