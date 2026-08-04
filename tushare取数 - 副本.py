"""

Tushare 免费版（500分）取数函数
覆盖：指数月K线（趋势信号用）
"""
import pandas as pd
from datetime import datetime, timedelta
import tushare as ts
import time
import os
from datetime import date

INDEX_TS_CODE  = "000300.SH"   # 沪深300
DATA_ROOT_1 = r'输出地址1'
DATA_ROOT_2 = r'输出地址2'
DATA_ROOT_3 = r'输出地址3'
DATA_ROOT_4 = r'输出地址4'

ts.set_token('*************************************')
pro = ts.pro_api()

def get_index(    ts_code: str = INDEX_TS_CODE,    years: int = 25) :
    """
    拉取指数月K线。

    返回列：trade_date, open, high, low, close, vol, amount
    trade_date 已转为 datetime 并设为索引，按时间升序排列。

    Parameters
    ----------
    ts_code : Tushare指数代码，默认沪深300
    years   : 拉取多少年历史，默认5年
    """
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")
    end_date   = datetime.today().strftime("%Y%m%d")

    df = pro.index_monthly (
        ts_code    = ts_code,
        start_date = start_date,
        end_date   = end_date,
        fields     = "trade_date,open,high,low,close,vol,amount"
    )

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df


def get_index_fixdate(dates: list[str], save_dir: str,
                       ts_code: str = '000905.SH',
                       sleep_interval: float = 0.3) -> pd.DataFrame:
    """
    按固定日期列表拉取中证500日K线（index_daily），
    模仿 fetch_price 风格，逐日循环取数后合并，输出 CSV。

    返回
    ----
    pd.DataFrame
        index=trade_date (datetime), columns=[open,high,low,close,vol,amount]
    """
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f'{ts_code.replace(".", "_")}_daily.csv')

    all_data: list[pd.DataFrame] = []
    skipped: list[str] = []

    for i, date in enumerate(dates):
        print(f"[{i + 1}/{len(dates)}] 尝试 {date} ...")
        try:
            df = pro.index_daily(
                ts_code    = ts_code,
                trade_date = date,
                fields     = "trade_date,open,high,low,close,vol,amount"
            )
            if not df.empty:
                all_data.append(df)
                print(f"  成功 {len(df)} 条")
            else:
                print("  跳过（非交易日或无数据）")
                skipped.append(date)
        except Exception as e:
            print(f"  出错: {e}")
            time.sleep(sleep_interval)
            continue
        time.sleep(sleep_interval)

    print(f"\n跳过的日期（共 {len(skipped)} 天）：")
    print(skipped)

    if not all_data:
        print("无数据，退出。")
        return None

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='%Y%m%d')
    df_all = df_all.sort_values('trade_date').set_index('trade_date')

    df_all.to_csv(output_file, index=True)
    print(f"\n取数完成，共 {len(df_all)} 个交易日")
    print(f"已保存至: {output_file}")
    print(df_all.head())
    return df_all


def fetch_price(dates: list[str],save_dir: str,field: str ,sleep_interval: float = 0.3,):

    os.makedirs(save_dir, exist_ok=True)
    filename = f'{field}_pivot.csv'   # 不传就自动命名
    output_file = os.path.join(save_dir, filename)

    all_data: list[pd.DataFrame] = []
    skipped_dates: list[str] = []

    for i, date in enumerate(dates):
        print(f"[{i + 1}/{len(dates)}] 尝试 {date} ...")
        try:
            df = pro.daily(trade_date=date, fields=f'ts_code,trade_date,{field}')  # ← 动态字段
            if not df.empty:
                all_data.append(df)
                print(f"  成功 {len(df)} 条")
            else:
                print("  跳过（非交易日或无数据）")
                skipped_dates.append(date)
        except Exception as e:
            print(f"  出错: {e}")
            time.sleep(sleep_interval)
            continue
        time.sleep(sleep_interval)

    print(f"\n跳过的日期（共 {len(skipped_dates)} 天）：")
    print(skipped_dates)

    if not all_data:
        print("无数据，退出。")
        return None

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='%Y%m%d')
    df_all.sort_values(['ts_code', 'trade_date'], inplace=True)

    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values=field)  # ← 动态字段
    df_pivot = df_pivot.sort_index(axis=1)

    df_pivot.to_csv(output_file, index=True)
    print(f"\nPivot 完成，已保存至 {output_file}")
    print(f"交易日数：{df_pivot.shape[0]}，股票数：{df_pivot.shape[1]}")
    print(df_pivot.iloc[:5, :5])

    return df_pivot

def fetch_adj_factor_pivot(dates: list[str],save_dir: str,output_filename: str = 'adj_factor_pivot.csv',sleep_interval: float = 0.3) -> pd.DataFrame | None:

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, output_filename)

    all_data: list[pd.DataFrame] = []
    skipped_dates: list[str] = []

    for i, date in enumerate(dates):
        print(f"[{i + 1}/{len(dates)}] 尝试 {date} ...")
        try:
            df = pro.adj_factor(trade_date=date, fields='ts_code,trade_date,adj_factor')
            if not df.empty:
                all_data.append(df)
                print(f"  成功 {len(df)} 条")
            else:
                print("  跳过（非交易日或无数据）")
                skipped_dates.append(date)
        except Exception as e:
            print(f"  出错: {e}")
            time.sleep(sleep_interval)
            continue
        time.sleep(sleep_interval)

    # ── 汇总跳过情况 ──────────────────────────────────────────
    print(f"\n跳过的日期（共 {len(skipped_dates)} 天）：")
    print(skipped_dates)

    if not all_data:
        print("无数据，退出。")
        return None

    # ── 合并 & Pivot ──────────────────────────────────────────
    df_all = pd.concat(all_data, ignore_index=True)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='%Y%m%d')
    df_all.sort_values(['ts_code', 'trade_date'], inplace=True)

    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values='adj_factor')
    df_pivot = df_pivot.sort_index(axis=1)

    df_pivot.to_csv(output_file, index=True)
    print(f"\nPivot 完成，已保存至 {output_file}")
    print(f"交易日数：{df_pivot.shape[0]}，股票数：{df_pivot.shape[1]}")
    print("前5行 × 前5列预览：")
    print(df_pivot.iloc[:5, :5])

    return df_pivot

def fetch_daily_basic(dates, field):
    """按月末交易日循环拉取，拼成宽表"""
    all_dfs = []
    for i, date in enumerate(dates):
        try:
            df = pro.daily_basic(
                trade_date=date,
                fields=f'ts_code,trade_date,{field}'
            )
            if df is not None and not df.empty:
                df = df.set_index('ts_code')[field].rename(date)
                all_dfs.append(df)
            print(f"[{i+1}/{len(dates)}] {date} OK，{len(df)} 条")
        except Exception as e:
            print(f"[{i+1}/{len(dates)}] {date} 出错: {e}")
        time.sleep(0.3)  # 2000积分200次/分钟，0.3s足够

    result = pd.DataFrame(all_dfs)          # shape: 日期 × 股票
    result.index = pd.to_datetime(result.index)
    return result

def fetch_industry():
    """拉申万2021二级行业分类，返回 ts_code → industry 映射"""
    # 先拿行业列表
    df_l2 = pro.index_classify(level='L3', src='SW2021')
    print(f"申万行业数：{len(df_l2)}")

    all_members = []
    for _, row in df_l2.iterrows():
        idx_code = row['index_code']
        idx_name = row['industry_name']
        try:
            df_member = pro.index_member(
                index_code=idx_code,
                fields='index_code,con_code,in_date,out_date,is_new'
            )
            df_member['industry_name'] = idx_name
            # 只保留当前成分股
            df_member = df_member[df_member['is_new'] == 'Y']
            all_members.append(df_member)
            print(f"  {idx_name}: {len(df_member)} 只")
        except Exception as e:
            print(f"  {idx_name} 出错: {e}")
        time.sleep(0.3)

    industry_df = pd.concat(all_members, ignore_index=True)
    industry_df = industry_df.rename(columns={'con_code': 'ts_code'})[
        ['ts_code', 'industry_name', 'index_code', 'in_date', 'out_date']
    ]
    return industry_df

def fetch_fina_indicator(codes, start_date='20150101', end_date='20260531'):
    all_dfs = []
    failed = []

    for i, code in enumerate(codes):
        try:
            df = pro.fina_indicator(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,ann_date,end_date,revenue_yoy,netprofit_yoy'
            )
            if df is not None and not df.empty:
                all_dfs.append(df)
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(codes)}] 进度 {(i+1)/len(codes)*100:.1f}%")
        except Exception as e:
            print(f"  {code} 出错: {e}，加入重试队列")
            failed.append(code)
            time.sleep(61)
        time.sleep(0.3)

    # 重试失败的
    if failed:
        print(f"\n重试 {len(failed)} 只...")
        for code in failed:
            try:
                df = pro.fina_indicator(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,revenue_yoy,netprofit_yoy'
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"  重试仍失败 {code}: {e}")
            time.sleep(0.5)

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=['ts_code', 'ann_date', 'end_date'])  # ← 加这行
    result = result.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)
    return result


def fetch_st_mask(stocks, dates) -> pd.DataFrame:
    """
    通过 namechange 表获取每只股票的历史名称变更记录，
    判断每个日期是否处于 ST / *ST 状态。
    返回 date×stock 的 bool DataFrame，True = 当日为ST。
    """
    st_cache_path = os.path.join(DATA_ROOT_2, "st_mask")
    print("    开始下载 namechange 数据（可能需要几分钟）...")
    records = []
    batch_size = 5

    for i in range(0, len(stocks), batch_size):
        batch = stocks[i: i + batch_size]
        for code in batch:
            try:
                df = pro.namechange(
                    ts_code=code,
                    fields="ts_code,name,start_date,end_date"
                )
                if df is not None and len(df) > 0:
                    records.append(df)
            except Exception as e:
                print(f"    警告: {code} namechange 失败 -> {e}")
            time.sleep(0.05)  # 限速

        print(f"    进度: {min(i + batch_size, len(stocks))}/{len(stocks)}")

    nc = pd.concat(records, ignore_index=True)
    print(nc)
    return nc

'''   # 构造 date×stock bool 矩阵
    is_st = pd.DataFrame(False, index=dates, columns=stocks)
    date_arr = pd.DatetimeIndex(dates)

    for _, row in nc.iterrows():
        code = row["ts_code"]
        name = str(row["name"]) if pd.notna(row["name"]) else ""
        if "ST" not in name.upper():
            continue
        if code not in is_st.columns:
            continue

        start = pd.to_datetime(str(row["start_date"])) if pd.notna(row["start_date"]) else date_arr[0]
        end = pd.to_datetime(str(row["end_date"])) if pd.notna(row["end_date"]) else date_arr[-1]

        mask_dates = date_arr[(date_arr >= start) & (date_arr <= end)]
        is_st.loc[mask_dates, code] = True

    is_st.to_parquet(st_cache_path)
    print(f"    ST mask 已缓存到 {st_cache_path}")'''

def fetch_all_sw3_members(pro, output_dir: str) -> pd.DataFrame:
    """
    拉取346个申万三级行业的全部历史成员记录，存本地备用。
    字段：con_code, index_code, industry_name, in_date, out_date

    注意：346个行业需要循环调用346次接口，建议存本地后不要重复拉取。
    """
    save_path = os.path.join(output_dir, 'sw3_members_all.csv')
    if os.path.exists(save_path):
        print(f">>> 从本地加载: {save_path}")
        df = pd.read_csv(save_path)
        df['in_date']  = pd.to_datetime(df['in_date'],  format='%Y%m%d', errors='coerce')
        df['out_date'] = pd.to_datetime(df['out_date'], format='%Y%m%d', errors='coerce')
        return df

    print(">>> 拉取申万三级行业列表...")
    sw3_list = pro.index_classify(level='L3', src='SW2021')
    print(f"    共 {len(sw3_list)} 个三级行业")

    all_records = []
    for i, row in sw3_list.iterrows():
        index_code = row['index_code']
        index_name = row['industry_name']
        try:
            members = pro.index_member(index_code=index_code,
                                       fields='con_code,index_code,in_date,out_date')
            members['industry_name'] = index_name
            all_records.append(members)
        except Exception as e:
            print(f"    [{i}] {index_name} 拉取失败: {e}")
        if i % 50 == 0:
            print(f"    进度: {i}/{len(sw3_list)}")

    df = pd.concat(all_records, ignore_index=True)
    df['in_date']  = pd.to_datetime(df['in_date'],  format='%Y%m%d', errors='coerce')
    df['out_date'] = pd.to_datetime(df['out_date'], format='%Y%m%d', errors='coerce')

    df.to_csv(save_path, index=False)
    print(f"    保存至: {save_path}, 共 {len(df)} 条记录")
    return df


if __name__ == "__main__":
    # ─────────── 拉取宽基指数价格数据 ─────────────
    #df = get_index()
    #df.to_excel(f'{DATA_ROOT_1}\\index_price.xlsx')  #宏观择时


    # ─────────── 拉取个股前复权因子数据 ──────────────
    # 从 Excel 读取完整日期列表
    trade_dates = pd.read_excel(os.path.join(DATA_ROOT_4, '2022-2025日期.xlsx'), header=None)[0].dt.strftime('%Y%m%d').tolist()
    today_str = date.today().strftime('%Y%m%d')
    trade_dates = [d for d in trade_dates if d <= today_str]
    # 查缺补漏时直接传入子列表
    #trade_dates = ['20210930']

    df_index_pivot= get_index_fixdate(dates=trade_dates, save_dir=DATA_ROOT_4, ts_code='000906.SH', sleep_interval=0.3)
    #df_adj_pivot = fetch_adj_factor_pivot(dates=trade_dates,save_dir=DATA_ROOT_4,output_filename='adj_factor_pivot.csv', )
    #df_close_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='close')
    #df_open_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='open')
    #df_low_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='low')
    #df_high_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='high')
    #df_amount_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='amount')
    #df_volume_pivot = fetch_price(dates=trade_dates, save_dir=DATA_ROOT_4, field='vol')


    # ─────────── 拉取个股估值因子数据 ──────────────
    #pe_df = fetch_daily_basic(trade_dates, field='pe_ttm')
    #pe_df.to_csv(os.path.join(DATA_ROOT_4, 'pe_ttm.csv'))
    #print(f"PE保存完成，shape: {pe_df.shape}")

    #pb_df = fetch_daily_basic(trade_dates, field='pb')
    #pb_df.to_csv(os.path.join(DATA_ROOT_4, 'pb.csv'))
    #print(f"PB保存完成，shape: {pb_df.shape}")

    # ─────────── 拉取个股市值数据 ──────────────
    #df_market_value_pivot = fetch_daily_basic(dates=trade_dates, field='total_mv')
    #df_market_value_pivot.to_csv(os.path.join(DATA_ROOT_4, 'market_value.csv'))
    #print(f"market_value保存完成，shape: {df_market_value_pivot.shape}")

    #df_stocks = pd.read_excel(os.path.join(DATA_ROOT_2, 'industry_sw2021.xlsx'), index_col=0)[:10]
    #all_codes = df_stocks.index.tolist()
    #fina_df = fetch_fina_indicator(codes=all_codes)
    #fina_df.to_excel(os.path.join(DATA_ROOT_2, 'fina_indicator_3.xlsx'), index=False)
    #print(f"财务指标保存完成，shape: {fina_df.shape}")

    # ─────────── 拉取个股行业分类数据 ──────────────
    #industry_df = fetch_industry()
    #industry_df.to_csv(os.path.join(DATA_ROOT_3, 'industry_sw2021.csv'), index=False)
    #print(f"行业分类保存完成，shape: {industry_df.shape}")

    # ─────────── 拉取个股ST数据 ──────────────
    #print(">>> 构建 ST mask...")
    #is_st = fetch_st_mask(all_codes, trade_dates)
    #print(f"    ST占比示例（最新日）: {is_st.iloc[-1].sum()} 只")


    #  拉取（或读取本地缓存的）全行业历史成员
    #members_df = fetch_all_sw3_members(pro, DATA_ROOT_3)
