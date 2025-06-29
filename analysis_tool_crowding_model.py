
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd


def corr_analysis(factor_index, integrated_score, future_days):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.font_manager as fm

    # 字體設定
    font_path = "/Users/yangzherui/Desktop/py coding/因子研究/STHeiti Light.ttc"
    font = fm.FontProperties(fname=font_path, size=10)

    # 協方差函數（交集填 0）
    def cov_with_intersection(a, b):
        idx = a.index.intersection(b.index)
        return a.loc[idx].fillna(0).corr(b.loc[idx].fillna(0))

    # 初始設定
    factors = factor_index.columns
    return_df = pd.DataFrame(index=factors)
    std_df = pd.DataFrame(index=factors)
    sharpe_df = pd.DataFrame(index=factors)

    # 計算各未來區間報酬、波動、Sharpe Ratio 與 integrated_score 的相關性
    for day in future_days:
        future_return = factor_index.shift(-day).rolling(day).mean()
        future_std = factor_index.shift(-day).rolling(day).std()
        future_sharpe = future_return / (future_std + 1e-6)  # 防除以 0

        return_df[day] = [
            cov_with_intersection(integrated_score[factor], future_return[factor]) for factor in factors
        ]
        std_df[day] = [
            cov_with_intersection(integrated_score[factor], future_std[factor]) for factor in factors
        ]
        sharpe_df[day] = [
            cov_with_intersection(integrated_score[factor], future_sharpe[factor]) for factor in factors
        ]

    # 漸層色生成器
    def generate_gradient_colors(base_color, n):
        cmap = mcolors.LinearSegmentedColormap.from_list("grad", ["white", base_color])
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    # 畫橫向 Grouped Bar 圖
    def plot_grouped_bar_horizontal(df, title, base_color):
        bar_height = 0.15
        y = np.arange(len(df.index))
        colors = generate_gradient_colors(base_color, len(future_days))

        plt.figure(figsize=(6, 6))
        for i, day in enumerate(future_days):
            plt.barh(
                y + i * bar_height, df[day],
                height=bar_height,
                color=colors[i],
                edgecolor='black',
                label=f'{day} days'
            )

        plt.yticks(
            y + bar_height * (len(future_days) - 1) / 2,
            df.index,
            fontproperties=font
        )
        plt.title(title, fontproperties=font)
        plt.xlabel('相關係數', fontproperties=font)
        plt.ylabel('因子', fontproperties=font)
        plt.legend(title='future_days', prop=font)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # 繪圖：報酬、波動、夏普
    plot_grouped_bar_horizontal(return_df, '與未來報酬的相關係數', base_color='skyblue')
    plot_grouped_bar_horizontal(std_df, '與未來波動的相關係數', base_color='lightgreen')
    plot_grouped_bar_horizontal(sharpe_df, '與未來夏普比率的相關係數', base_color='orange')



def analysis_crowding_factor(integrated_score,date):
    data = integrated_score.loc[date]
    # 字體設定（本機字體路徑）
    font_path = "/Users/yangzherui/Desktop/py coding/因子研究/STHeiti Light.ttc"
    font_prop = fm.FontProperties(fname=font_path, size=10)
    # 排序後繪圖
    sorted_items = sorted(data.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_items)
    plt.figure(figsize=(8, 14))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel("值", fontproperties=font_prop)
    plt.ylabel("指標", fontproperties=font_prop)
    plt.title("各指標值長條圖", fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def mdd_analysis(integrated_score,factor_index):
    def compute_max_drawdown(cum_returns):
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def compute_dd_exceed_ratio(factor_return_df, crowding_bool_df, dd_threshold=0.1, forward_days=252):
        ratios = []
        for factor in factor_return_df.columns:
            returns = factor_return_df[factor]
            crowding = crowding_bool_df[factor]
            dd_count = 0
            total_true = 0
            for date in returns.index:
                if date not in crowding.index or not crowding.loc[date]:
                    continue
                idx_loc = returns.index.get_loc(date)
                if idx_loc + forward_days >= len(returns):
                    continue
                forward_returns = returns.iloc[idx_loc:idx_loc + forward_days]
                cum_returns = (1 + forward_returns).cumprod()
                max_dd = compute_max_drawdown(cum_returns)
                if max_dd <= -dd_threshold:
                    dd_count += 1
                total_true += 1
            if total_true > 0:
                ratios.append(dd_count / total_true)
        return np.mean(ratios) if ratios else np.nan

    one=(integrated_score<=-1)
    two=(integrated_score>-1)&(integrated_score<-0.5)
    three=(integrated_score>-0.5)&(integrated_score<0)
    four=(integrated_score>0)&(integrated_score<0.5)
    five=(integrated_score>0)&(integrated_score<0.5)
    six=(integrated_score>=1)

    future_dd_df = pd.DataFrame({
        'less_than_-1': [compute_dd_exceed_ratio(factor_index, one)],
        '-1~-0.5': [compute_dd_exceed_ratio(factor_index, two)],
        '-0.5~-0': [compute_dd_exceed_ratio(factor_index, three)],
        '0~0.5': [compute_dd_exceed_ratio(factor_index, four)],
        '0.5~1': [compute_dd_exceed_ratio(factor_index, five)],
        'greater_than_1': [compute_dd_exceed_ratio(factor_index, six)],
    })
    # 字體設定（使用本機字體）
    font_path = "/Users/yangzherui/Desktop/py coding/因子研究/STHeiti Light.ttc"
    font_prop = fm.FontProperties(fname=font_path, size=10)

    # 資料（從 DataFrame 中取出 row 0）
    data = future_dd_df.iloc[0]
    #colors = ['green', 'gray', 'red']  # uncrowding, neutral, crowding

    plt.figure(figsize=(8, 5))
    bars = plt.bar(data.index, data.values)#, color=colors

    # 數值標籤
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}',
                ha='center', va='bottom', fontsize=10, fontproperties=font_prop)

    plt.title('未來一年出現超過5%最大回撤的機率（依綜合擁擠評分分類）', fontproperties=font_prop, fontsize=14)
    plt.ylabel('風險機率', fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.ylim(0, max(data.values)*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def future_month_return(integrated_score,factor_index):
    uncrowding=(integrated_score<=-0.5)
    neutral=(integrated_score>-0.5)&(integrated_score<0.5)
    crowding=(integrated_score>=0.5)

    combined_result = {
        'month': [],
        'uncrowding': [],
        'neutral': [],
        'crowding': []
    }


    for n in range(1,25):
        future_cum_return = factor_index.shift(n * 20* -1).rolling(20).mean()
        uncrowding_avg = future_cum_return[uncrowding].mean(axis=1).mean()
        neutral_avg = future_cum_return[neutral].mean(axis=1).mean()
        crowding_avg = future_cum_return[crowding].mean(axis=1).mean()
        combined_result['month'].append(f'{n}_month')
        combined_result['uncrowding'].append(uncrowding_avg)
        combined_result['neutral'].append(neutral_avg)
        combined_result['crowding'].append(crowding_avg)

    future_return_df = pd.DataFrame(combined_result)

    # 假設你的 future_return_df 已經存在，且包含 'month'、'uncrowding'、'neutral'、'crowding'

    plt.figure(figsize=(12, 6))
    plt.plot(future_return_df['month'], future_return_df['uncrowding'], label='Uncrowding', marker='o')
    plt.plot(future_return_df['month'], future_return_df['neutral'], label='Neutral', marker='o')
    plt.plot(future_return_df['month'], future_return_df['crowding'], label='Crowding', marker='o')

    # 圖表美化
    plt.xlabel('Future Month')
    plt.ylabel('Average monthly Return (%)')
    plt.title('Uncrowding vs Neutral vs Crowding\nFuture monthly Returns (1-23 months)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def future_month_volitity(integrated_score,factor_index):
    uncrowding=(integrated_score<=-0.5)
    neutral=(integrated_score>-0.5)&(integrated_score<0.5)
    crowding=(integrated_score>=0.5)

    combined_result = {
        'month': [],
        'uncrowding': [],
        'neutral': [],
        'crowding': []
    }


    for n in range(1, 25):
        future_cum_return = factor_index.shift(n * 20*-1).rolling(20).std()
        uncrowding_avg = future_cum_return[uncrowding].mean(axis=1).mean()
        neutral_avg = future_cum_return[neutral].mean(axis=1).mean()
        crowding_avg = future_cum_return[crowding].mean(axis=1).mean()
        combined_result['month'].append(f'{n}_month')
        combined_result['uncrowding'].append(uncrowding_avg)
        combined_result['neutral'].append(neutral_avg)
        combined_result['crowding'].append(crowding_avg)

    future_return_df = pd.DataFrame(combined_result)


    plt.figure(figsize=(12, 6))
    plt.plot(future_return_df['month'], future_return_df['uncrowding'], label='Uncrowding', marker='o')
    plt.plot(future_return_df['month'], future_return_df['neutral'], label='Neutral', marker='o')
    plt.plot(future_return_df['month'], future_return_df['crowding'], label='Crowding', marker='o')

    # 圖表美化
    plt.xlabel('Future Month')
    plt.ylabel('Average monthly std')
    plt.title('Uncrowding vs Neutral vs Crowding\nFuture monthly std (1-23 months)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def future_month_sharpe(integrated_score, factor_index):
    uncrowding = (integrated_score <= -0.5)
    neutral = (integrated_score > -0.5) & (integrated_score < 0.5)
    crowding = (integrated_score >= 0.5)

    combined_result = {
        'month': [],
        'uncrowding': [],
        'neutral': [],
        'crowding': []
    }

    for n in range(1, 25):
        future_return = factor_index.shift(n * 20 * -1).rolling(20).mean()
        future_std = factor_index.shift(n * 20 * -1).rolling(20).std()
        future_sharpe = future_return / (future_std + 1e-6)  # 防止除以 0

        uncrowding_avg = future_sharpe[uncrowding].mean(axis=1).mean()
        neutral_avg = future_sharpe[neutral].mean(axis=1).mean()
        crowding_avg = future_sharpe[crowding].mean(axis=1).mean()

        combined_result['month'].append(f'{n}_month')
        combined_result['uncrowding'].append(uncrowding_avg)
        combined_result['neutral'].append(neutral_avg)
        combined_result['crowding'].append(crowding_avg)

    future_sharpe_df = pd.DataFrame(combined_result)

    plt.figure(figsize=(12, 6))
    plt.plot(future_sharpe_df['month'], future_sharpe_df['uncrowding'], label='Uncrowding', color='orange', marker='o')
    plt.plot(future_sharpe_df['month'], future_sharpe_df['neutral'], label='Neutral', color='gray', marker='o')
    plt.plot(future_sharpe_df['month'], future_sharpe_df['crowding'], label='Crowding', color='tomato', marker='o')

    # 圖表美化
    plt.xlabel('Future Month')
    plt.ylabel('Average monthly Sharpe ratio')
    plt.title('Uncrowding vs Neutral vs Crowding\nFuture monthly Sharpe ratio (1-23 months)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

