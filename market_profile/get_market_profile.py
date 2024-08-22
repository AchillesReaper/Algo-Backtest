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


def get_data_for_mp(underlying:Underlying) -> pd.DataFrame:
    '''
    This is custom function to get the custom market data for constructing daily market profil.
    1. get 30 min data for the underlying
    2. combining: </br> 
        a. the rows btw 17:00 to 21:29 into one </br>
        b. the rows btw 21:30 to 02:30 into one </br>
    3. return the cleaned dataFrame { index:<i>datetime</i>, columns:<i>['open', 'high', 'low', 'close', 'volume', 'expiry', 'trade_date']</i>}
    '''
    underlying.barSizeSetting = IBBarSize.MIN_30
    df_raw = get_spot_future_ib(underlying)
    df_raw = df_raw[['datetime', 'open', 'high', 'low', 'close', 'volume', 'expiry', 'trade_date']]
    df_raw.set_index('datetime', inplace=True)

    df_clean = df_raw.iloc[0:0]
    for td in df_raw['trade_date'].unique():
        df_td = df_raw[df_raw['trade_date'] == td]        
        # combin the rows btw 17:00 to 21:29 into one
        df_td = combine_rows(df_td, '17:00', '21:29')
        # combin the rows btw 21:30 to 02:30 into one
        df_td = combine_rows(df_td, '21:30', '02:30')
        df_clean = df_clean._append(df_td)

    return df_clean


def gen_market_profile(df_clean:pd.DataFrame)->pd.DataFrame:
    '''
    This is custom function to transform 30min data into daily market profile.
    Return a df with columns: ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'skewness', 'kurtosis', 'val', 'vah', 'pocs', 'tpo_count']
    '''

    trade_date_list     = df_clean["trade_date"].unique()
    tpo_count_list      = []
    pocs_list           = []
    val_list            = []
    vah_list            = []
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
        td_tpo_dist = {}
        price_record = []
        for index, row in df_td.iterrows():
            for price in range(row["low"], row["high"] + 1):
                # 5 points per TPO
                if price % 5 == 0:
                    if price not in td_tpo_dist:
                        td_tpo_dist[price] = 1
                    elif price in td_tpo_dist:
                        td_tpo_dist[price] += 1
                    # for value area calculation later
                    price_record.append(price)
        td_tpo_dist = {
            k: v for k, v in sorted(td_tpo_dist.items(), key=lambda item: item[0])
        }
        tpo_count_list.append(td_tpo_dist)

        # for pocs[], VAL[], VAH[], and skewness[]
        if len(td_tpo_dist) > 0:
            td_poc_count = max(td_tpo_dist.values())
            td_poc = [
                price for price, count in td_tpo_dist.items() if count == td_poc_count
            ]

            tpo_stdev = int(np.std(price_record, ddof=1))
            mode_price = int(np.mean(td_poc))

            td_skew = round(skew(price_record), 4)
            td_kurt = round(kurtosis(price_record), 4)

            pocs_list.append(td_poc)
            val_list.append(mode_price - tpo_stdev)
            vah_list.append(mode_price + tpo_stdev)
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
            "pocs"      : pocs_list,
            "tpo_count" : tpo_count_list,
        },
        index=trade_date_list,
    )
    df_mp.index.name = 'trade_date'
    return df_mp

if __name__ == "__main__":
    underlying = Underlying(
        symbol='HSI',
        exchange='HKFE',
        contract_type='FUT',
        barSizeSetting=IBBarSize.DAY_1,
        start_date='2024-01-01',
        end_date='2024-01-31',
    )

    df_clean = get_data_for_mp(underlying)
    print(df_clean.head())
    df_mp = gen_market_profile(df_clean)
    print(df_mp.head())