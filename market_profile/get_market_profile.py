import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from dh_backtest.models.data_classes import Underlying, IBBarSize
from dh_backtest.models.remote_data import get_spot_future_ib


def combine_rows(df_td:pd.DataFrame, start_time:str, end_time:str) -> pd.DataFrame:
    mask_start = pd.to_datetime(start_time).time()
    mask_end = pd.to_datetime(end_time).time()
    mask = df_td.between_time(mask_start, mask_end).copy()

    if not mask.empty:
        df_td = df_td.drop(mask.index)
        df_td.loc[mask.index[0]] = {
            'open'      : mask['open'].iloc[0],
            'high'      : mask['high'].max(),
            'low'       : mask['low'].min(),
            'close'     : mask['close'].iloc[-1],
            'volume'    : mask['volume'].sum(),
            'expiry'    : mask['expiry'].iloc[0],
            'trade_date': mask['trade_date'].iloc[0],
        }

    return df_td


def get_data_for_mp(underlying:Underlying, folder_path:str, resolution:str= IBBarSize.MIN_30) -> pd.DataFrame:
    '''
    parameter resolution: '30min' or '1day'
    This is custom function to get the custom market data for constructing daily market profil.
    1. get 30 min data for the underlying
    2. combining: </br> 
        a. the rows btw 17:00 to 21:29 into one </br>
        b. the rows btw 21:30 to 02:30 into one </br>
    3. return the cleaned dataFrame { index:<i>datetime</i>, columns:<i>['open', 'high', 'low', 'close', 'volume', 'expiry', 'trade_date']</i>}
    '''
    underlying.barSizeSetting = resolution
    df_raw = get_spot_future_ib(underlying)
    df_raw = df_raw[['datetime', 'open', 'high', 'low', 'close', 'volume', 'expiry', 'trade_date']]
    df_raw.set_index('datetime', inplace=True)

    df_clean = df_raw.iloc[0:0]
    for td in df_raw['trade_date'].unique():
        df_td = df_raw[df_raw['trade_date'] == td]        
        # combin the rows btw 17:00 to 21:29 into one
        df_td = combine_rows(df_td, '17:00', '21:29')
        # combin the rows btw 21:30 to 03:30 into one
        df_td = combine_rows(df_td, '21:30', '03:30')
        df_clean = df_clean._append(df_td)

    file_name = f'{underlying.symbol}_clean_{underlying.start_date}_{underlying.end_date}.csv'.replace('-','')
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    df_clean.to_csv(f'{folder_path}/{file_name}', index=True)
    return df_clean


def gen_market_profile(folder_path:str, underlying:Underlying, resolution:str= IBBarSize.MIN_30, is_update_data:bool=True)->pd.DataFrame:
    '''
    This is custom function to transform 30min data into daily market profile.
    Return a df with columns: ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'skewness', 'kurtosis', 'val', 'vah', 'pocs', 'tpo_count']
    '''
    # get df_clean from local or api
    if is_update_data:
        df_clean = get_data_for_mp(underlying, folder_path=folder_path, resolution=resolution)
    else:
        file_name = f'{underlying.symbol}_clean_{underlying.start_date}_{underlying.end_date}.csv'.replace('-','')
        df_clean = pd.read_csv(f'{folder_path}/{file_name}', index_col=0)

    trade_date_list     = df_clean["trade_date"].unique()
    tpo_count_list      = []
    pocs_list           = []
    val_list            = []
    vah_list            = []
    spike_upper_list    = []
    spike_lower_list    = []
    skewness_list       = []
    kurtosis_list       = []
    open_price_list     = []
    high_price_list     = []
    low_price_list      = []
    close_price_list    = []
    volume_list         = []
    for td in trade_date_list:
        df_td = df_clean[df_clean["trade_date"] == td]
        # summarize the trade date data
        open_price_list.append(df_td["open"].iloc[0])
        high_price_list.append(df_td["high"].max())
        low_price_list.append(df_td["low"].min())
        close_price_list.append(df_td["close"].iloc[-1])
        volume_list.append(df_td["volume"].sum())

        # calculate the market profile varibales
        td_tpo_dict = {}
        price_record = []
        
        td_tpo_dict_start   = (df_td['low'].min()/5).__ceil__() * 5
        td_tpo_dict_end     = (df_td['high'].max()/5).__ceil__() * 5
        for tag in range(td_tpo_dict_start, td_tpo_dict_end+1, 5):
            td_tpo_dict[tag] = 0
        for index, row in df_td.iterrows():
            price_record += list(range(row['low'], row['high']+1))
            for tag in td_tpo_dict:
                if tag >= row['low'] and tag <= row['high']+4:
                    td_tpo_dict[tag] += 1

        td_tpo_dict = {
            k: v for k, v in sorted(td_tpo_dict.items(), key=lambda item: item[0])
        }
        tpo_count_list.append(td_tpo_dict)

        # for pocs[], VAL[], VAH[], and skewness[]
        if len(td_tpo_dict) > 0:
            td_poc_count = max(td_tpo_dict.values())
            td_poc = [
                price for price, count in td_tpo_dict.items() if count == td_poc_count
            ]

            tpo_stdev   = int(np.std(price_record, ddof=1))
            mode_price  = int(np.mean(td_poc))
            val         = mode_price - tpo_stdev
            vah         = mode_price + tpo_stdev

            spkl = td_tpo_dict_start
            for price in td_tpo_dict.keys():
                if (price < val) & (td_tpo_dict[price] <= 2):
                    spkl = price
                else:
                    break

            spkh = td_tpo_dict_end
            for price in sorted(td_tpo_dict.keys(), reverse=True):
                if (price > vah) & (td_tpo_dict[price] <=2):
                    spkh = price
                else:
                    break

            td_skew = round(skew(price_record), 4)
            td_kurt = round(kurtosis(price_record), 4)

            pocs_list.append(td_poc)
            val_list.append(val)
            vah_list.append(vah)
            spike_lower_list.append(spkl)
            spike_upper_list.append(spkh)
            skewness_list.append(td_skew)
            kurtosis_list.append(td_kurt)
        else:
            pocs_list.append([])
            val_list.append(None)
            vah_list.append(None)
            skewness_list.append(None)
            kurtosis_list.append(None)

    df_mp = pd.DataFrame(
        {
            "open"      : open_price_list,
            "high"      : high_price_list,
            "low"       : low_price_list,
            "close"     : close_price_list,
            "volume"    : volume_list,
            "skewness"  : skewness_list,
            "kurtosis"  : kurtosis_list,
            "val"       : val_list,
            "vah"       : vah_list,
            "spkl"      : spike_lower_list,
            "spkh"      : spike_upper_list,
            "pocs"      : pocs_list,
            "tpo_count" : tpo_count_list,
        },
        index=trade_date_list,
    )
    df_mp.index.name = 'trade_date'

    file_name = f'{underlying.symbol}_mp_{underlying.start_date}_{underlying.end_date}.csv'.replace('-','')
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    df_mp.to_csv(f'{folder_path}/{file_name}', index=True)
    return df_mp

if __name__ == "__main__":
    underlying = Underlying(
        symbol='HSI',
        exchange='HKFE',
        contract_type='FUT',
        barSizeSetting=IBBarSize.MIN_5,
        start_date='2024-01-01',
        end_date='2024-01-31',
    )
    is_update_data = True
    folder_path = 'data/market_profile'

    # df_clean = get_data_for_mp(underlying, IBBarSize.MIN_15)
    # print(df_clean.head())
    df_mp = gen_market_profile(underlying, is_update_data=is_update_data, folder_path=folder_path)
    print(df_mp.head())