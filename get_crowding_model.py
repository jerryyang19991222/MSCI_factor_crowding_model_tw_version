import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

input_path = input("請輸入輸入檔案路徑 (.pkl)：") or '/Users/yangzherui/Desktop/py coding/因子研究/data/cmoney_eqlw_twse.pkl'
output_path = input("請輸入輸出資料夾路徑：") or 'handler/'
factor_return_type = input("請輸入 factor return 計算方式 (quantile_ls / top_bottom / IC 等)：") or 'quantile_ls'

print("\n✅ 輸入參數如下：")
print(f"📥 輸入檔案：{input_path}")
print(f"📤 輸出資料夾：{output_path}")
print(f"🧠 因子報酬類型：{factor_return_type}")

data=pd.read_csv('/Users/yangzherui/Desktop/py coding/因子研究/data/APIPRCD.csv')###導入業師提供資料
data['mdate'] = pd.to_datetime(data['mdate'])###時間資料轉為時間格式
data.rename(columns= {'coid': 'symbol','mdate': 'datetime'}, inplace=True)###時間與股票代碼對其其她資料名稱
# 過濾出屬於 TWSE 且 symbol 為 4 碼數字的資料
TWSE_data = data[(data['mkt'] == 'TWSE') &(data['symbol'].astype(str).str.len() == 4) &(data['symbol'].astype(str).str.isdigit())]
TWSE_data = TWSE_data[(TWSE_data['symbol'].astype(int) >= 1101) &(TWSE_data['symbol'].astype(int) <= 9999)]
TWSE_data=TWSE_data.set_index(['datetime','symbol'])

####需要探討proxy
short_data=pd.read_csv('/Users/yangzherui/Desktop/py coding/因子研究/data/日融資券排行.csv')###導入業師提供資料
short_data = short_data[short_data['日期'].notna()]
short_data['日期'] = pd.to_datetime(short_data['日期'].astype(int).astype(str), format='%Y%m%d')
short_data['股票代號']=short_data['股票代號'].astype(str)
short_data.rename(columns= {'股票代號': 'symbol','日期': 'datetime'}, inplace=True)###時間與股票代碼對其其她資料名稱
short_TWSE_data = short_data[(short_data['symbol'].astype(str).str.len() == 4) &(short_data['symbol'].astype(str).str.isdigit())]
short_TWSE_data = short_TWSE_data[(short_TWSE_data['symbol'].astype(int) >= 1101) &(short_TWSE_data['symbol'].astype(int) <= 9999)]
short_TWSE_data=short_TWSE_data.set_index(['datetime','symbol'])

comoney_actual=pd.read_pickle(input_path)

def get(columns):
    return TWSE_data[columns].unstack()

def get_short(columns):
    return short_TWSE_data[columns].unstack().reindex(columns=get('pbr_tej').columns,index=get('pbr_tej').index).replace(0, np.nan)

def get_cactual(columns):
    return comoney_actual[columns].unstack()

market_data = pd.read_excel("/Users/yangzherui/Desktop/py coding/因子研究/data/twse.xlsx", skiprows=4)  # 視情況更改 skiprows 數字
del market_data['TWA00 加權指數']
market_data['日期'] = pd.to_datetime(market_data['日期'])
market_data.rename(columns= {'日期': 'datetime'}, inplace=True)###時間與股票代碼對其其她資料名稱
market_data=market_data.set_index(['datetime'])
market_data.sort_index(inplace=True)###時間排序
market_return=((market_data['收盤價']/market_data['收盤價'].shift(1))-1)


def factor_return(factor,factor_return_type ='quantile_ls',quantile_num=5,period='D'):
    if factor_return_type=='quantile_ls':
        ###quantile_ls
        long_position=(factor.rank(axis=1, pct=True, ascending=True)>((int(quantile_num)-1)/int(quantile_num))).astype(float)
        short_position=(factor.rank(axis=1, pct=True, ascending=True)<(1/int(quantile_num))).astype(float)
        long_position=long_position.div(long_position.sum(axis=1),axis=0)
        short_position=short_position.div(short_position.sum(axis=1),axis=0)
        ls_position=long_position-short_position
        factor_return=((ls_position*(((get('open_d')*get('adjfac')).shift(-2) / (get('open_d')*get('adjfac')).shift(-1)) - 1)).sum(axis=1)).shift(2)###處理前視偏誤
    if factor_return_type=='IC':
        ###IC_SE
        factor_return=factor.corrwith((((get('open_d')*get('adjfac')).shift(-20) / (get('open_d')*get('adjfac'))) - 1),method='spearman',axis = 1).shift(20)###處理前視偏誤
        factor_return=factor_return#.rolling(252).mean()
    if factor_return_type=='only_long':
        long_position=(factor.rank(axis=1, pct=True, ascending=True)>((int(quantile_num)-1)/int(quantile_num))).astype(float)
        long_position=long_position.div(long_position.sum(axis=1),axis=0)
        factor_return=((long_position*(((get('open_d')*get('adjfac')).shift(-2) / (get('open_d')*get('adjfac')).shift(-1)) - 1)).sum(axis=1)).shift(2)#
    return factor_return

print('計算因子收益率中')


factor_index=pd.DataFrame()
###計算指數因子收益率
for columns in [col for col in comoney_actual.columns if col not in ['Unnamed: 0','股票名稱']]:
    factor_index[columns]=factor_return(factor=get_cactual(columns),factor_return_type =factor_return_type,quantile_num=5)
    print(f'======================={columns}完成=========================')

factor_index.to_pickle(f'{output_path}factor_index_{factor_return_type}.pkl')

print('計算估值價差中')


def valuation_spread(factor, valuation):
    # 計算分位數排名（按橫向，每個時間點個股的因子值排名）
    rank = factor.rank(axis=1, pct=True)
    # 挑出 Top 20% 和 Bottom 20%
    top_quantile = valuation[rank > 0.8].median(axis=1)
    bottom_quantile = valuation[rank < 0.2].median(axis=1)
    raw_spread = bottom_quantile / top_quantile
    expanding_mean = raw_spread.expanding(min_periods=30).mean()
    expanding_std = raw_spread.expanding(min_periods=30).std()
    z_score = (raw_spread - expanding_mean) / expanding_std
    return z_score

def valuation_spread_for_ep(factor, valuation):
    rank = factor.rank(axis=1, pct=True)
    top_quantile = valuation[rank > 0.8].median(axis=1)
    bottom_quantile = valuation[rank < 0.2].median(axis=1)
    raw_spread = bottom_quantile - top_quantile
    expanding_mean = raw_spread.expanding(min_periods=30).mean()
    expanding_std = raw_spread.expanding(min_periods=30).std()
    z_score = (raw_spread - expanding_mean) / expanding_std
    return z_score

###直接分四個
book_to_price=pd.DataFrame()
earning_to_price=pd.DataFrame()
sales_to_price=pd.DataFrame()
integrate_score=pd.DataFrame()

for columns in [col for col in factor_index.columns if col not in ['Unnamed: 0','股票名稱']]:
    book_to_price[columns]=valuation_spread(get_cactual(columns),(np.log(1+(1 / get('pbr_tej')))))
    earning_to_price[columns]=valuation_spread(get_cactual(columns),(1 / get('per_tej')))
    sales_to_price[columns]=valuation_spread(get_cactual(columns),(1 / get('psr_tej')))
    integrate_score[columns] = pd.concat([book_to_price[columns],earning_to_price[columns],sales_to_price[columns]], axis=1).mean(axis=1, skipna=True)#sales_to_price[columns]
    print(f'======================={columns}完成=========================')

integrate_score.to_pickle(f'{output_path}valuation_spread_{factor_return_type}.pkl')



print('計算因子反轉中')


def get_weight(factor_index):
    weighting=factor_index.rolling(252*5).sum()
    global_std = weighting.expanding(min_periods=30).std().mean(axis=1)
    demean= weighting - weighting.expanding(min_periods=30).mean() 
    return demean.div(global_std, axis=0)

factor_reversal_=get_weight(factor_index)

factor_reversal_.to_pickle(f'{output_path}factor_reversal_{factor_return_type}.pkl')


print('計算因子波動度中')


def factor_volatility(factor_return,market_return=market_return,std_rolling_period=63):
    factor_volatility=(factor_return.rolling(std_rolling_period).std())/((market_return.rolling(std_rolling_period).std()))
    return factor_volatility

weighting=pd.DataFrame()
for columns in [col for col in factor_index.columns if col not in ['Unnamed: 0','股票名稱']]:
    weighting[columns]=factor_volatility(factor_return=factor_index[columns],market_return=market_return,std_rolling_period=63)
    print(f'======================={columns}完成=========================')

def get_weight(factor_volatility):
    return (factor_volatility-factor_volatility.expanding(min_periods=30).mean())/(factor_volatility.expanding(min_periods=30).std())

factor_volatility=get_weight(weighting)

factor_volatility.to_pickle(f'{output_path}factor_volatility_{factor_return_type}.pkl')


print('計算成對相關係數中')


####計算預期收益率
調整後開盤價=(get('open_d')*get('adjfac')).shift(-1)
預期收益率＿日=(調整後開盤價.shift(-1)/調整後開盤價)-1

def pairwise_corr(factor, quantile_num=5, corr_period=63):
    from tqdm import tqdm

    single_stock_return = (get('close_d') / get('close_d').shift(1)) - 1
    top_corrs = {}
    bottom_corrs = {}
    window = corr_period
    dates = single_stock_return.index.sort_values()

    def calc_group_corr_fast(df):
        # 移除變異數太小的股票（避免除以 0）
        stds = df.std()
        valid_columns = stds[stds > 1e-8].index
        df = df[valid_columns]
        corrs = []
        for stock in df.columns:
            others = df.drop(columns=stock)
            mean_others = others.mean(axis=1)
            if df[stock].isna().all() or mean_others.isna().all():
                continue
            corr = df[stock].corr(mean_others)
            if pd.notna(corr):
                corrs.append(corr)

        return np.nanmean(corrs) if corrs else np.nan

    for current_date in tqdm(dates[window - 1:], desc="Calculating pairwise_corr"):
        current_idx = dates.get_loc(current_date)
        window_dates = dates[current_idx - window + 1 : current_idx + 1]

        if current_date not in factor.index:
            continue

        todays_factor = factor.loc[current_date]
        pct_rank = todays_factor.rank(pct=True, ascending=True)
        top_condition = pct_rank > ((quantile_num - 1) / quantile_num)
        bottom_condition = pct_rank < (1 / quantile_num)

        top_stocks = pct_rank.index[top_condition]
        bottom_stocks = pct_rank.index[bottom_condition]

        valid_stocks = single_stock_return.columns
        top_stocks = [s for s in top_stocks if s in valid_stocks]
        bottom_stocks = [s for s in bottom_stocks if s in valid_stocks]

        top_window_returns = single_stock_return.loc[window_dates, top_stocks]
        bottom_window_returns = single_stock_return.loc[window_dates, bottom_stocks]

        # 補值處理：用每支股票自己的平均補值
        top_window_returns = top_window_returns.apply(lambda x: x.fillna(x.mean()), axis=0)
        bottom_window_returns = bottom_window_returns.apply(lambda x: x.fillna(x.mean()), axis=0)

        top_corr = calc_group_corr_fast(top_window_returns)
        bottom_corr = calc_group_corr_fast(bottom_window_returns)

        top_corrs[current_date] = top_corr
        bottom_corrs[current_date] = bottom_corr

    pairwise_corr = (pd.Series(top_corrs) + pd.Series(bottom_corrs)) / 2
    pairwise_corr = pairwise_corr#.dropna()
    return pairwise_corr

def get_weight(weighting):  
    global_std = weighting.expanding(min_periods=30).std().mean(axis=1)
    demean= weighting - weighting.expanding(min_periods=30).mean() 
    return demean.div(global_std, axis=0)

weighting=pd.DataFrame()
for columns in [col for col in factor_index.columns if col not in ['Unnamed: 0','股票名稱']]:
    weighting[columns]=pairwise_corr(get_cactual(columns),quantile_num=5,corr_period=63)
    print(f'======================={columns}完成=========================')

pairwise_corr_=get_weight(weighting)

pairwise_corr_.to_pickle(f'{output_path}pairwise_corr_{factor_return_type}.pkl')


print('計算放空未平倉量中')


def get_quantile_indicators(factor_df, quantile_num=5,factor_name='mom'):
    rank_pct = factor_df.rank(axis=1, ascending=True, pct=True)
    indicators = {}
    for q in range(quantile_num):
        lower = q / quantile_num
        upper = (q + 1) / quantile_num
        indicator = ((rank_pct > lower) & (rank_pct <= upper)).astype(int)
        indicators[f'{factor_name}_Q{q+1}'] = indicator

    return pd.concat(indicators,axis=1)

def rolling_regression_by_date(df, y_col, x_cols, window=63):
    from tqdm import tqdm
    import statsmodels.api as sm
    df = df.sort_index(level=0)
    dates = df.index.get_level_values(0).unique()
    betas = []
    for i in tqdm(range(window - 1, len(dates)), desc="Rolling Regression"):
        date_window = dates[i - window + 1 : i + 1]
        window_df = df.loc[date_window]
        y = window_df[y_col]
        X = window_df[x_cols]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            betas.append(model.params)
        except:
            betas.append(pd.Series([float('nan')] * (len(x_cols)+1), index=['const'] + x_cols))
    result_index = dates[window - 1:]
    beta_df = pd.DataFrame(betas, index=result_index)
    return beta_df

def get_quantile_indicators(factor_df, quantile_num=5,factor_name='mom'):
    rank_pct = factor_df.rank(axis=1, ascending=True, pct=True)
    indicators = {}
    for q in range(quantile_num):
        lower = q / quantile_num
        upper = (q + 1) / quantile_num
        indicator = ((rank_pct > lower) & (rank_pct <= upper)).astype(int)
        indicators[f'{factor_name}_Q{q+1}'] = indicator

    return pd.concat(indicators,axis=1)


def rolling_regression_by_date(df, y_col, x_cols, window=63):
    from tqdm import tqdm
    import statsmodels.api as sm
    df = df.sort_index(level=0)
    dates = df.index.get_level_values(0).unique()
    betas = []
    for i in tqdm(range(window - 1, len(dates)), desc="Rolling Regression"):
        date_window = dates[i - window + 1 : i + 1]
        window_df = df.loc[date_window]
        y = window_df[y_col]
        X = window_df[x_cols]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            betas.append(model.params)
        except:
            betas.append(pd.Series([float('nan')] * (len(x_cols)+1), index=['const'] + x_cols))
    result_index = dates[window - 1:]
    beta_df = pd.DataFrame(betas, index=result_index)
    return beta_df

def short_interest_spread(factor,quantile_num=5,beta_window=63,short_interest=(get_short('借券賣出餘額')/(get_short('借券可使用額度')))):
    ###計算其他factor
    mom_factor=((get('close_d')*get('adjfac')).shift(20*1)/(get('close_d')*get('adjfac')).shift(20*4)-1)
    value_factor=(1/get('pbr_tej'))
    size_factor=get('mktcap')
    ###計算滾動回歸所需要之panel_data
    factor_indicate=get_quantile_indicators(factor, quantile_num=quantile_num,factor_name='factor')
    mom_factor_indicate=get_quantile_indicators(mom_factor, quantile_num=quantile_num,factor_name='mom_factor')
    value_factor_indicate=get_quantile_indicators(value_factor, quantile_num=quantile_num,factor_name='value_factor')
    size_factor_indicate=get_quantile_indicators(size_factor, quantile_num=quantile_num,factor_name='size_factor')
    short_interest=short_interest#get_short('借券賣出')#(get_short('券餘')/(get_short('借券可使用額度'))).reindex(columns=get('pbr_tej').columns,index=get('pbr_tej').index)#防止無限(也可以0變NA)
    combine_df=pd.concat([factor_indicate.stack(),mom_factor_indicate.stack(),value_factor_indicate.stack(),size_factor_indicate.stack(),short_interest.stack()],axis=1)#.fillna(0)
    del combine_df['factor_Q3']
    del combine_df['mom_factor_Q3']
    del combine_df['value_factor_Q3']
    del combine_df['size_factor_Q3']
    ###執行滾動回歸
    rolling_beta=rolling_regression_by_date(combine_df,0, [col for col in combine_df.columns if col not in [0]], window=beta_window)
    short_interest_spread=rolling_beta[f'factor_Q1']-rolling_beta[f'factor_Q{quantile_num}']
    return short_interest_spread

def get_weight(weighting):  
    global_std = weighting.expanding(min_periods=30).std().mean(axis=1)
    demean= weighting - weighting.expanding(min_periods=30).mean() 
    return demean.div(global_std, axis=0)

weighting=pd.DataFrame()
for columns in [col for col in factor_index.columns if col not in ['Unnamed: 0','股票名稱']]:
    weighting[columns]=short_interest_spread(factor=get_cactual(columns),quantile_num=5,beta_window=63)
    print(f'======================={columns}完成=========================')

short_interest_spread=get_weight(weighting.loc['2008':])

short_interest_spread.to_pickle(f'{output_path}short_interest_spread_{factor_return_type}.pkl')


print('計算綜合擁擠指標中')


dfs = [valuation_spread,pairwise_corr,factor_volatility,factor_reversal_,short_interest_spread]
combined = pd.concat(dfs, axis=0, keys=range(len(dfs)))
integrated_score = combined.groupby(level=1).mean()
integrated_score.to_pickle(f'{output_path}crowdind_score_{factor_return_type}.pkl')





