import pandas as pd
import numpy as np
import multiprocessing as mp
import tqdm
from config import config

windows_ver1 = [3, 5, 10, 15, 20, 30]
windows_ver2 = [9, 12, 26, 60]
windows_ver3 = [1, 2, 3, 5, 7, 10, 13, 15, 20]
windows_ver4 = [1] + windows_ver1

def compute_cross_sectional_features(df, value_col, date_col='日期'):
    """
    对 DataFrame 的 value_col 列，按日期分组计算：
    - 截面 Rank / 当天股票总数   (值域 [0,1])
    - 截面 Z-Score               (均值为0，标准差为1)
    直接在 df 上添加两列：'{value_col}_rank' 和 '{value_col}_zscore'
    返回新增的列名元组。
    """
    rank_col = f'{value_col}_rank'
    zscore_col = f'{value_col}_zscore'

    # 每个截面上的股票数
    count_per_date = df.groupby(date_col)[value_col].transform('count')

    # 排名（1～N）
    df[rank_col] = df.groupby(date_col)[value_col].rank(pct=False) / count_per_date

    # Z-Score
    mean_per_date = df.groupby(date_col)[value_col].transform('mean')
    std_per_date = df.groupby(date_col)[value_col].transform('std')
    df[zscore_col] = (df[value_col] - mean_per_date) / std_per_date.replace(0, np.nan)
    df[zscore_col].fillna(0, inplace=True)

    return rank_col, zscore_col

# 特征工程
def engineer_features_plusversion(df):
    """
    计算158个Alpha特征和39个技术指标特征，并安全合并。
    """
    df_copy = df.copy().reset_index(drop=True)   # 重置索引，确保对齐

    # 1. 计算158个Alpha特征
    df_alpha = engineer_features_alpha(df_copy)
    # 2. 计算39个技术指标特征
    df_tech = engineer_features(df_copy)

    # 3. 获取技术指标列名（不含原始基础列）
    tech_cols = get_tech_feature_names()
    existing_tech_cols = [col for col in tech_cols if col in df_tech.columns]

    # 4. 从 Alpha 表中剔除与技术指标重复的基础列（保留原始列只保留一份）
    #    通常 Alpha 表已包含所有原始列，技术指标表只取新增特征列
    base_cols_in_158 = [col for col in df_alpha.columns if col not in existing_tech_cols]
    df_158_selected = df_alpha[base_cols_in_158]

    # 5. 合并（索引已重置，直接按位置拼接）
    df_final = pd.concat([df_158_selected, df_tech[existing_tech_cols]], axis=1)

    # 6. 去重（安全措施）
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # 7. 处理无穷值与缺失值
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    return df_final

def engineer_features(df):
    """
    生成技术指标特征（不含全局截面特征）：
    - 价格幅度波动
    - 布林带中轨/标准差
    - EMA
    - MACD快线/信号线
    - 成交量均线 / 比例 / 变化率
    - OBV变动
    - 波动率
    - KDJ
    - ATR
    - RSI
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 1. 价格幅度波动 (4个)
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 2. 布林带中轨 & 标准差 (windows_ver1)
    for w in windows_ver1:
        df[f'boll_mid_{w}'] = talib.SMA(close, timeperiod=w)
        df[f'boll_std_{w}'] = talib.STDDEV(close, timeperiod=w, nbdev=1)

    # 3. EMA (windows_ver2)
    for w in windows_ver2:
        df[f'ema_{w}'] = talib.EMA(close, timeperiod=w)

    # 4. MACD 快线及信号线 (相邻窗口对)
    for i in range(len(windows_ver2) - 1):
        w1 = windows_ver2[i]
        w2 = windows_ver2[i + 1]
        macd_line, signal_line, _ = talib.MACD(close, fastperiod=w1, slowperiod=w2, signalperiod=9)
        df[f'macd_{w1}_{w2}'] = macd_line
        df[f'macd_signal_{w1}_{w2}'] = signal_line

    # 5. 成交量均线 (windows_ver1)
    for w in windows_ver1:
        df[f'volume_ma_{w}'] = talib.SMA(volume, timeperiod=w)

    # 6. 成交量比例 (相邻窗口对)
    for i in range(len(windows_ver1) - 1):
        w1 = windows_ver1[i]
        w2 = windows_ver1[i + 1]
        df[f'volume_ratio_{w1}_{w2}'] = df[f'volume_ma_{w1}'] / df[f'volume_ma_{w2}']

    # 7. 成交量变化率 (windows_ver3)
    for w in windows_ver3:
        df[f'volume_change_{w}'] = volume.pct_change(periods=w)

    # 8. OBV 及其多窗口变动
    obv = talib.OBV(close, volume)
    df['obv'] = obv
    for w in windows_ver3:
        df[f'obv_change_{w}'] = obv - obv.shift(w)

    # 9. 波动率 (日收益率滚动标准差, windows_ver1)
    ret_1 = close.pct_change(1)
    for w in windows_ver1:
        df[f'volatility_{w}'] = ret_1.rolling(w).std()

    # 10. KDJ (windows_ver2)
    for w in windows_ver2:
        k, d = talib.STOCH(high, low, close, fastk_period=w, slowk_period=3, slowd_period=3)
        df[f'kdj_k_{w}'] = k
        df[f'kdj_d_{w}'] = d
        df[f'kdj_j_{w}'] = 3 * k - 2 * d

    # 11. ATR (windows_ver2)
    for w in windows_ver2:
        df[f'atr_{w}'] = talib.ATR(high, low, close, timeperiod=w)

    # 12. RSI (windows_ver2)
    for w in windows_ver2:
        df[f'rsi_{w}'] = talib.RSI(close, timeperiod=w)

    # 清理无穷值与缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def engineer_features_alpha(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    global windows_ver1, windows_ver3

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows_ver3:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows_ver1:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows_ver1:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15 features) - 使用 talib 加速
    for w in windows_ver1:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')

        # R-squared can be calculated as CORREL^2
        rsquare = close.rolling(w).apply(
            lambda x: np.corrcoef(x, np.arange(w))[0, 1] ** 2,
            raw=True
        )
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows_ver1:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows_ver1:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows_ver1:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows_ver1:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows_ver1:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows_ver1:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15 features) - talib 不支持，保留原实现
    for w in windows_ver1:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows_ver1:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows_ver1:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows_ver1:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')

    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows_ver1:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15 features) - 向量化操作，无需更改
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows_ver1:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows_ver1:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows_ver1:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15 features) - 向量化操作，无需更改
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows_ver1:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows_ver1:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows_ver1:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows_ver1:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows_ver1:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows_ver1:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15 features) - 向量化操作，无需更改
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows_ver1:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows_ver1:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows_ver1:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names

    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)

    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def add_global_cross_sectional_features(df, date_col='日期'):
    """
    在全量股票数据（含多只股票、多日期）上计算以下特征的截面 Rank 和 Z-Score：
    - 涨跌幅 (ret_k)
    - 波动率 (volatility_k)
    - 日内振幅 (avg_amplitude_k)
    - 平均成交量 (avg_volume_k)
    - 平均换手率 (avg_turnover_k)
    - 参考成本偏离度 CGO
    """
    df = df.copy()
    # 确保按股票和日期排序
    df = df.sort_values(['股票代码', date_col]).reset_index(drop=True)

    # 定义需要计算的中间量及其窗口
    windows = windows_ver4   # [1, 3, 5, 10, 15, 20, 30]

    # 1. 为每只股票计算各种滚动统计量（中间列，以下划线开头）
    def _calc_stock_rolling(grp):
        grp = grp.sort_values(date_col).copy()
        c = grp['收盘'].astype(float)
        v = grp['成交量'].astype(float)
        h = grp['最高'].astype(float)
        l = grp['最低'].astype(float)
        turnover = grp['换手率'].astype(float)

        # 涨跌幅 ret_k
        for k in windows:
            grp[f'_ret_{k}'] = c.pct_change(periods=k)

        # 波动率 volatility_k（日收益率的标准差）
        ret_daily = c.pct_change(1)
        for k in windows:
            grp[f'_volatility_{k}'] = ret_daily.rolling(k).std()

        # 日内振幅 avg_amplitude_k
        amplitude = (h - l) / c.shift(1)
        for k in windows:
            grp[f'_avg_amplitude_{k}'] = amplitude.rolling(k).mean()

        # 平均成交量 avg_volume_k
        for k in windows:
            grp[f'_avg_volume_{k}'] = v.rolling(k).mean()

        # 平均换手率 avg_turnover_k
        for k in windows:
            grp[f'_avg_turnover_{k}'] = turnover.rolling(k).mean()

        # CGO: 参考成本价 = 过去100日换手率加权成交均价
        vwap = grp['成交额'] / (v + 1e-12)
        weighted_price = vwap * turnover
        sum_wp = weighted_price.rolling(100, min_periods=1).sum()
        sum_turn = turnover.rolling(100, min_periods=1).sum()
        ref_cost = sum_wp / (sum_turn + 1e-12)
        grp['_CGO'] = (c - ref_cost) / (ref_cost + 1e-12)
        return grp

    df = df.groupby('股票代码', group_keys=False).apply(_calc_stock_rolling)

    # 2. 按日期计算截面 Rank 和 Z-Score
    for k in windows:
        compute_cross_sectional_features(df, f'_ret_{k}', date_col=date_col)
        compute_cross_sectional_features(df, f'_volatility_{k}', date_col=date_col)
        compute_cross_sectional_features(df, f'_avg_amplitude_{k}', date_col=date_col)
        compute_cross_sectional_features(df, f'_avg_volume_{k}', date_col=date_col)
        compute_cross_sectional_features(df, f'_avg_turnover_{k}', date_col=date_col)
    compute_cross_sectional_features(df, '_CGO', date_col=date_col)

    # 3. 重命名为最终列名（去掉下划线前缀）
    rename_dict = {}
    for k in windows:
        rename_dict[f'_ret_{k}_rank'] = f'ret_{k}_rank'
        rename_dict[f'_ret_{k}_zscore'] = f'ret_{k}_zscore'
        rename_dict[f'_volatility_{k}_rank'] = f'volatility_{k}_rank'
        rename_dict[f'_volatility_{k}_zscore'] = f'volatility_{k}_zscore'
        rename_dict[f'_avg_amplitude_{k}_rank'] = f'avg_amplitude_{k}_rank'
        rename_dict[f'_avg_amplitude_{k}_zscore'] = f'avg_amplitude_{k}_zscore'
        rename_dict[f'_avg_volume_{k}_rank'] = f'avg_volume_{k}_rank'
        rename_dict[f'_avg_volume_{k}_zscore'] = f'avg_volume_{k}_zscore'
        rename_dict[f'_avg_turnover_{k}_rank'] = f'avg_turnover_{k}_rank'
        rename_dict[f'_avg_turnover_{k}_zscore'] = f'avg_turnover_{k}_zscore'
    rename_dict['_CGO_rank'] = 'CGO_rank'
    rename_dict['_CGO_zscore'] = 'CGO_zscore'
    df.rename(columns=rename_dict, inplace=True)

    # 删除临时中间列
    cols_to_drop = [col for col in df.columns if col.startswith('_')]
    df.drop(columns=cols_to_drop, inplace=True)

    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_tech_feature_names():
    """
    返回 engineer_features_39 实际生成的全部技术指标的列名。
    （不含原始列，如 '开盘'、'收盘' 等）
    """
    global windows_ver1, windows_ver2, windows_ver3, windows_ver4

    tech = []

    # 价格幅度波动
    tech.extend([
        'high_low_spread', 'open_close_spread',
        'high_close_spread', 'low_close_spread'
    ])

    # 布林带中轨 & 标准差
    for w in windows_ver1:
        tech.append(f'boll_mid_{w}')
        tech.append(f'boll_std_{w}')

    # EMA
    for w in windows_ver2:
        tech.append(f'ema_{w}')

    # MACD 快线 & 信号线 (相邻窗口对)
    for i in range(len(windows_ver2) - 1):
        w1 = windows_ver2[i]
        w2 = windows_ver2[i + 1]
        tech.append(f'macd_{w1}_{w2}')
        tech.append(f'macd_signal_{w1}_{w2}')

    # 成交量均线
    for w in windows_ver1:
        tech.append(f'volume_ma_{w}')

    # 成交量比例
    for i in range(len(windows_ver1) - 1):
        w1 = windows_ver1[i]
        w2 = windows_ver1[i + 1]
        tech.append(f'volume_ratio_{w1}_{w2}')

    # 成交量变化率
    for w in windows_ver3:
        tech.append(f'volume_change_{w}')

    # OBV & OBV 变化
    tech.append('obv')
    for w in windows_ver3:
        tech.append(f'obv_change_{w}')

    # 波动率 (日收益率滚动标准差)
    for w in windows_ver1:
        tech.append(f'volatility_{w}')

    # KDJ
    for w in windows_ver2:
        tech.append(f'kdj_k_{w}')
        tech.append(f'kdj_d_{w}')
        tech.append(f'kdj_j_{w}')

    # ATR
    for w in windows_ver2:
        tech.append(f'atr_{w}')

    # RSI
    for w in windows_ver2:
        tech.append(f'rsi_{w}')

    return tech

def get_global_feature_names():
    """
    返回 engineer_features_39 实际生成的全部全局特征的列名。
    （不含原始列，如 '开盘'、'收盘' 等）
    """
    global windows_ver1, windows_ver2, windows_ver3, windows_ver4

    # 全局截面特征
    global_tech = []
    for k in windows_ver4:
        global_tech.append(f'ret_{k}_rank')
        global_tech.append(f'ret_{k}_zscore')
        global_tech.append(f'volatility_{k}_rank')
        global_tech.append(f'volatility_{k}_zscore')
        global_tech.append(f'avg_amplitude_{k}_rank')
        global_tech.append(f'avg_amplitude_{k}_zscore')
        global_tech.append(f'avg_volume_{k}_rank')
        global_tech.append(f'avg_volume_{k}_zscore')
        global_tech.append(f'avg_turnover_{k}_rank')
        global_tech.append(f'avg_turnover_{k}_zscore')
    global_tech.extend(['CGO_rank', 'CGO_zscore'])
    return global_tech

def init_feature_columns_map():
    """
    根据所有定义的窗口，动态生成 feature_columns_map。
    返回字典，包含 '39' 和 '158+39' 两个键。
    复用 get_tech39_feature_names() 以消除重复代码。
    """
    # 窗口定义（仅用于 Alpha 因子）
    global  windows_ver1, windows_ver3

    # 基础列（原始数据 + 日频衍生）
    base_cols = [
        'instrument', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
        '振幅', '涨跌额', '换手率', '涨跌幅', '市盈率' ,'市销率' ,'市现率' ,'市净率' ,'交易状态', '是否ST'
    ]

    # 技术指标 + 全局特征（直接复用，无需重复定义）
    tech_and_global = get_tech_feature_names() + get_global_feature_names()

    # ========== Alpha 因子部分（来自 engineer_features）==========
    alpha_cols = []

    # K线形态 (9)
    alpha_cols.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])
    # 价格相对 (4)
    alpha_cols.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    # ROC (windows_ver3)
    for w in windows_ver3:
        alpha_cols.append(f'ROC{w}')

    # MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV,
    # IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD,
    # VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD (全部使用 windows_ver1)
    for w in windows_ver1:
        alpha_cols.append(f'MA{w}')
        alpha_cols.append(f'STD{w}')
        alpha_cols.append(f'BETA{w}')
        alpha_cols.append(f'RSQR{w}')
        alpha_cols.append(f'RESI{w}')

    for w in windows_ver1:
        alpha_cols.append(f'MAX{w}')
    for w in windows_ver1:
        alpha_cols.append(f'MIN{w}')

    for w in windows_ver1:
        alpha_cols.append(f'QTLU{w}')
        alpha_cols.append(f'QTLD{w}')

    for w in windows_ver1:
        alpha_cols.append(f'RANK{w}')
        alpha_cols.append(f'RSV{w}')

    for w in windows_ver1:
        alpha_cols.append(f'IMAX{w}')
        alpha_cols.append(f'IMIN{w}')
        alpha_cols.append(f'IMXD{w}')

    for w in windows_ver1:
        alpha_cols.append(f'CORR{w}')
        alpha_cols.append(f'CORD{w}')

    for w in windows_ver1:
        alpha_cols.append(f'CNTP{w}')
        alpha_cols.append(f'CNTN{w}')
        alpha_cols.append(f'CNTD{w}')

    for w in windows_ver1:
        alpha_cols.append(f'SUMP{w}')
        alpha_cols.append(f'SUMN{w}')
        alpha_cols.append(f'SUMD{w}')

    for w in windows_ver1:
        alpha_cols.append(f'VMA{w}')
        alpha_cols.append(f'VSTD{w}')
        alpha_cols.append(f'WVMA{w}')

    for w in windows_ver1:
        alpha_cols.append(f'VSUMP{w}')
        alpha_cols.append(f'VSUMN{w}')
        alpha_cols.append(f'VSUMD{w}')

    # 组合两种模式
    feature_39 = base_cols + tech_and_global
    feature_158_39 = base_cols + alpha_cols + tech_and_global

    return {'39': feature_39, '158+39': feature_158_39}

feature_cloums_map = init_feature_columns_map()

feature_engineer_func_map = {
    '39': engineer_features,
    '158+39': engineer_features_plusversion
}

def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed

def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]

    # 1. 复制并转换日期格式为 datetime，确保排序正确
    df = df.copy()
    # 将日期列转换为 datetime 类型（适配 '2024/1/1' 等格式）
    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
    # 丢弃日期无效的行（如有）
    df = df.dropna(subset=['日期'])
    # 严格按股票代码和日期升序排序
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}（不含全局截面特征）...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    # 多进程分别计算每只股票的技术指标特征（无全局截面）
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 2. 映射股票索引
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # 3. 添加全局截面特征（此时所有股票数据已合并）
    print("正在添加全局截面特征（Rank / Z-Score）...")
    processed = add_global_cross_sectional_features(processed, date_col='日期')

    # 4. 构建标签并清洗
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

    # 5. 将日期列转换为统一字符串格式（便于保存和查看）
    processed['日期'] = processed['日期'].dt.strftime('%Y-%m-%d')

    return processed, feature_cloums_map[config['feature_num']]   # 特征列名保持不变

# 数据预处理函数(训练集、验证集)
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)

def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)