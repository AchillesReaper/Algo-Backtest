import copy
import json
import sys
from dh_backtest.models.data_classes import Underlying
from dh_backtest.models.remote_data import get_stock_futu_api
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import os
from futu import KLType
from termcolor import cprint



class MarketProfileGenerator():
    def __init__(
        self,
        folder_path:str,
        underlying:Underlying,
        is_update_data:bool=True,
        get_raw_data:callable=get_stock_futu_api
    ) -> None:
        self.folder_path    = folder_path
        self.underlying     = underlying
        self.is_update_data = is_update_data
        self.get_raw_data   = get_raw_data
        self.file_name_clean_data = f'{self.underlying.symbol}_clean_{self.underlying.start_date}_{self.underlying.end_date}.csv'.replace('-','')
        self.file_name_mp = f'{self.underlying.symbol}_mp_{self.underlying.start_date}_{self.underlying.end_date}.csv'.replace('-','')


    def get_tick_size(self, price:float) -> float:
        if price >= 0.01 and price < 1:
            return 0.001
        elif price >= 1 and price < 5:
            return 0.002
        elif price >= 5 and price < 10:
            return 0.005
        elif price >= 10 and price < 20:
            return 0.01
        elif price >= 20 and price < 100:
            return 0.02
        elif price >= 100 and price < 200:
            return 0.05
        elif price >= 200 and price < 500:
            return 0.1
        elif price >= 500 and price < 1000:
            return 0.2
        elif price >= 1000 and price < 2000:
            return 0.5
        else:
            return 1


    def get_data_for_mp(self) -> pd.DataFrame:
        '''
        This function is designed for getting stock data from futuapi
        1. get 30 min data for the underlying
        3. return the cleaned dataFrame { index:<i>datetime</i>, columns:<i>['open', 'high', 'low', 'close', 'volume', 'expiry', 'trade_date']</i>}
        '''
        if self.is_update_data:
            df_clean = self.get_raw_data(self.underlying)[['code', 'name', 'time_key', 'open', 'high', 'low', 'close', 'volume']]
            df_clean['trade_date'] = df_clean['time_key'].apply(lambda x: x.split(' ')[0])
            
            if not os.path.exists(self.folder_path): os.makedirs(self.folder_path)
            df_clean.to_csv(f'{self.folder_path}/{self.file_name_clean_data}', index=False)
            cprint(f'{self.file_name_clean_data} has been saved to {self.folder_path}', 'green')
        else:
            df_clean = pd.read_csv(f'{self.folder_path}/{self.file_name_clean_data}')
        return df_clean


    def gen_market_profile(self) -> pd.DataFrame:
        '''
        This function is designed for generating market profile from cleaned data
        '''
        df_clean = self.get_data_for_mp()

        trade_date_list = df_clean['trade_date'].unique()
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
            df_td = df_clean[df_clean['trade_date'] == td]
            # summarize the trade date data
            open_price_list.append(df_td['open'].iloc[0])
            high_price_list.append(df_td['high'].max())
            low_price_list.append(df_td['low'].min())
            close_price_list.append(df_td['close'].iloc[-1])
            volume_list.append(df_td['volume'].sum())

            # calculate the market profile varibales
            td_tpo_dict     = {}
            price_record    = []
            ticker_size     = self.get_tick_size(df_td['low'].min())
            bin_size        = ticker_size * 5

            td_tpo_dict_start   = float((df_td['low'].min()//bin_size) * bin_size)
            td_tpo_dict_end     = float((df_td['high'].max()//bin_size) * bin_size)

            # for tag in range(td_tpo_dict_start, td_tpo_dict_end+ticker_size, bin_size):
            #     td_tpo_dict[tag] = 0
            tag = copy.deepcopy(td_tpo_dict_start)
            while tag <= td_tpo_dict_end:
                td_tpo_dict[tag] = 0
                tag += bin_size

            for index, row in df_td.iterrows():
                price = copy.deepcopy(row['low'])
                while price <= row['high']:
                    price_record.append(price)
                    price += ticker_size
                for tag in td_tpo_dict:
                    if tag >= row['low'] and tag < row['high']+ticker_size:
                        td_tpo_dict[tag] += 1


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
                    "trade_date": trade_date_list,
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
                }
            )
        df_mp['code'] = self.underlying.symbol
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        df_mp.to_csv(f'{folder_path}/{self.file_name_mp}', index=False)
        cprint(f'{self.file_name_mp} has been saved to {folder_path}', 'green')
        return df_mp



if __name__ == "__main__":
    stock_list = [
        # "HK.00388", 
        # "HK.02888", "HK.00005", "HK.02388", "HK.00011", 
        # "HK.03988", "HK.01398", "HK.00939", "HK.01288", 
        # "HK.01299", "HK.02628", 
        "HK.02318",
        # "HK.00700", 
        # "HK.00857", "HK.00386", "HK.00883",
    ]
    start_date      = "2016-01-01"
    end_date        = "2024-07-31"

    for stock in stock_list:
        underlying = Underlying(
            symbol          = stock,
            barSizeSetting  = KLType.K_30M,
            start_date      = start_date,
            end_date        = end_date,
        )
        folder_path = 'market_profile/shape_training/data'
        is_update_data = False
        MarketProfileGenerator(folder_path=folder_path, underlying=underlying, is_update_data=is_update_data).gen_market_profile()


