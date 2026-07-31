import pandas as pd
import numpy as np

'''
国泰君安 Alpha191（《基于短周期价量特征的多因子选股体系》）
公式全部基于日频价量数据，不涉及财务/基本面。

原始公式里的基础变量与本项目 dfs 字典的对应关系：
    OPEN / HIGH / LOW / CLOSE / VOLUME  -> dfs["open"/"high"/"low"/"close"/"volume"]
    VWAP                                -> dfs["vwap"]（本项目用 AMOUNT/VOLUME 近似）
    RET（个股日收益率）                  -> close.pct_change()，公式内部按需现算，不单独建字段
    benchmarkindexOPEN / benchmarkindexCLOSE
        -> 基准指数开盘价/收盘价，本项目 dfs 字典目前未提供；
           前20个因子均不需要，用到时再接入 tushare index_daily。

原始公式里出现的 MAX(X, N) / MIN(X, N)（第二个参数是纯数字周期）一律按滚动窗口
（TSMAX/TSMIN）实现，与 MAX(A, B)（两个同形状序列的逐元素取大）区分开。
'''

# ── 注册表：每个alpha需要哪些df ──────────────────────────────
ALPHA_REGISTRY = {
    "alpha001": {"func": lambda dfs: alpha001(dfs["close"], dfs["open"], dfs["volume"]), "required": ["close", "open", "volume"]},
    "alpha002": {"func": lambda dfs: alpha002(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha003": {"func": lambda dfs: alpha003(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha004": {"func": lambda dfs: alpha004(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha005": {"func": lambda dfs: alpha005(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha006": {"func": lambda dfs: alpha006(dfs["open"], dfs["high"]), "required": ["open", "high"]},
    "alpha007": {"func": lambda dfs: alpha007(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha008": {"func": lambda dfs: alpha008(dfs["high"], dfs["low"], dfs["vwap"]), "required": ["high", "low", "vwap"]},
    "alpha009": {"func": lambda dfs: alpha009(dfs["high"], dfs["low"], dfs["volume"]), "required": ["high", "low", "volume"]},
    "alpha010": {"func": lambda dfs: alpha010(dfs["close"]), "required": ["close"]},
    "alpha011": {"func": lambda dfs: alpha011(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha012": {"func": lambda dfs: alpha012(dfs["open"], dfs["close"], dfs["vwap"]), "required": ["open", "close", "vwap"]},
    "alpha013": {"func": lambda dfs: alpha013(dfs["high"], dfs["low"], dfs["vwap"]), "required": ["high", "low", "vwap"]},
    "alpha014": {"func": lambda dfs: alpha014(dfs["close"]), "required": ["close"]},
    "alpha015": {"func": lambda dfs: alpha015(dfs["open"], dfs["close"]), "required": ["open", "close"]},
    "alpha016": {"func": lambda dfs: alpha016(dfs["volume"], dfs["vwap"]), "required": ["volume", "vwap"]},
    "alpha017": {"func": lambda dfs: alpha017(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha018": {"func": lambda dfs: alpha018(dfs["close"]), "required": ["close"]},
    "alpha019": {"func": lambda dfs: alpha019(dfs["close"]), "required": ["close"]},
    "alpha020": {"func": lambda dfs: alpha020(dfs["close"]), "required": ["close"]},
    "alpha021": {"func": lambda dfs: alpha021(dfs["close"]), "required": ["close"]},
    "alpha022": {"func": lambda dfs: alpha022(dfs["close"]), "required": ["close"]},
    "alpha023": {"func": lambda dfs: alpha023(dfs["close"]), "required": ["close"]},
    "alpha024": {"func": lambda dfs: alpha024(dfs["close"]), "required": ["close"]},
    "alpha025": {"func": lambda dfs: alpha025(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha026": {"func": lambda dfs: alpha026(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha027": {"func": lambda dfs: alpha027(dfs["close"]), "required": ["close"]},
    "alpha028": {"func": lambda dfs: alpha028(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha029": {"func": lambda dfs: alpha029(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha030": {"func": lambda dfs: alpha030(dfs),"required": ["close", "cap", "pe", "benchmarkindex", "universe_snapshots"]},
    "alpha031": {"func": lambda dfs: alpha031(dfs["close"]), "required": ["close"]},
    "alpha032": {"func": lambda dfs: alpha032(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha033": {"func": lambda dfs: alpha033(dfs["close"], dfs["low"], dfs["volume"]), "required": ["close", "low", "volume"]},
    "alpha034": {"func": lambda dfs: alpha034(dfs["close"]), "required": ["close"]},
    "alpha035": {"func": lambda dfs: alpha035(dfs["open"], dfs["volume"]), "required": ["open", "volume"]},
    "alpha036": {"func": lambda dfs: alpha036(dfs["volume"], dfs["vwap"]), "required": ["volume", "vwap"]},
    "alpha037": {"func": lambda dfs: alpha037(dfs["open"], dfs["close"]), "required": ["open", "close"]},
    "alpha038": {"func": lambda dfs: alpha038(dfs["high"]), "required": ["high"]},
    "alpha039": {"func": lambda dfs: alpha039(dfs["close"], dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["close", "open", "vwap", "volume"]},
    "alpha040": {"func": lambda dfs: alpha040(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha041": {"func": lambda dfs: alpha041(dfs["vwap"]), "required": ["vwap"]},
    "alpha042": {"func": lambda dfs: alpha042(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha043": {"func": lambda dfs: alpha043(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha044": {"func": lambda dfs: alpha044(dfs["low"], dfs["volume"], dfs["vwap"]), "required": ["low", "volume", "vwap"]},
    "alpha045": {"func": lambda dfs: alpha045(dfs["close"], dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["close", "open", "vwap", "volume"]},
    "alpha046": {"func": lambda dfs: alpha046(dfs["close"]), "required": ["close"]},
    "alpha047": {"func": lambda dfs: alpha047(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha048": {"func": lambda dfs: alpha048(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha049": {"func": lambda dfs: alpha049(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha050": {"func": lambda dfs: alpha050(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha051": {"func": lambda dfs: alpha051(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha052": {"func": lambda dfs: alpha052(dfs["high"], dfs["low"], dfs["close"]), "required": ["high", "low", "close"]},
    "alpha053": {"func": lambda dfs: alpha053(dfs["close"]), "required": ["close"]},
    "alpha054": {"func": lambda dfs: alpha054(dfs["close"], dfs["open"]), "required": ["close", "open"]},
    "alpha055": {"func": lambda dfs: alpha055(dfs["close"], dfs["open"], dfs["high"], dfs["low"]), "required": ["close", "open", "high", "low"]},
    "alpha056": {"func": lambda dfs: alpha056(dfs["open"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["open", "high", "low", "volume"]},
    "alpha057": {"func": lambda dfs: alpha057(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha058": {"func": lambda dfs: alpha058(dfs["close"]), "required": ["close"]},
    "alpha059": {"func": lambda dfs: alpha059(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha060": {"func": lambda dfs: alpha060(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha061": {"func": lambda dfs: alpha061(dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["low", "vwap", "volume"]},
    "alpha062": {"func": lambda dfs: alpha062(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha063": {"func": lambda dfs: alpha063(dfs["close"]), "required": ["close"]},
    "alpha064": {"func": lambda dfs: alpha064(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha065": {"func": lambda dfs: alpha065(dfs["close"]), "required": ["close"]},
    "alpha066": {"func": lambda dfs: alpha066(dfs["close"]), "required": ["close"]},
    "alpha067": {"func": lambda dfs: alpha067(dfs["close"]), "required": ["close"]},
    "alpha068": {"func": lambda dfs: alpha068(dfs["high"], dfs["low"], dfs["volume"]), "required": ["high", "low", "volume"]},
    "alpha069": {"func": lambda dfs: alpha069(dfs["open"], dfs["high"], dfs["low"]), "required": ["open", "high", "low"]},
    "alpha070": {"func": lambda dfs: alpha070(dfs["amount"]), "required": ["amount"]},
    "alpha071": {"func": lambda dfs: alpha071(dfs["close"]), "required": ["close"]},
    "alpha072": {"func": lambda dfs: alpha072(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha073": {"func": lambda dfs: alpha073(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha074": {"func": lambda dfs: alpha074(dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["low", "vwap", "volume"]},
    "alpha075": {"func": lambda dfs: alpha075(dfs["open"], dfs["close"], dfs["benchmarkindex"]), "required": ["close", "open", "benchmarkindex"]},
    "alpha076": {"func": lambda dfs: alpha076(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha077": {"func": lambda dfs: alpha077(dfs["high"], dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["high", "low", "vwap", "volume"]},
    "alpha078": {"func": lambda dfs: alpha078(dfs["high"], dfs["low"], dfs["close"]), "required": ["high", "low", "close"]},
    "alpha079": {"func": lambda dfs: alpha079(dfs["close"]), "required": ["close"]},
    "alpha080": {"func": lambda dfs: alpha080(dfs["volume"]), "required": ["volume"]},
    "alpha081": {"func": lambda dfs: alpha081(dfs["volume"]), "required": ["volume"]},
    "alpha082": {"func": lambda dfs: alpha082(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha083": {"func": lambda dfs: alpha083(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha084": {"func": lambda dfs: alpha084(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha085": {"func": lambda dfs: alpha085(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha086": {"func": lambda dfs: alpha086(dfs["close"]), "required": ["close"]},
    "alpha087": {"func": lambda dfs: alpha087(dfs["open"], dfs["high"], dfs["low"], dfs["vwap"]), "required": ["open", "high", "low", "vwap"]},
    "alpha088": {"func": lambda dfs: alpha088(dfs["close"]), "required": ["close"]},
    "alpha089": {"func": lambda dfs: alpha089(dfs["close"]), "required": ["close"]},
    "alpha090": {"func": lambda dfs: alpha090(dfs["volume"], dfs["vwap"]), "required": ["volume", "vwap"]},
    "alpha091": {"func": lambda dfs: alpha091(dfs["close"], dfs["low"], dfs["volume"]), "required": ["close", "low", "volume"]},
    "alpha092": {"func": lambda dfs: alpha092(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha093": {"func": lambda dfs: alpha093(dfs["open"], dfs["high"], dfs["low"]), "required": ["open", "high", "low"]},
    "alpha094": {"func": lambda dfs: alpha094(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha095": {"func": lambda dfs: alpha095(dfs["amount"]), "required": ["amount"]},
    "alpha096": {"func": lambda dfs: alpha096(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha097": {"func": lambda dfs: alpha097(dfs["volume"]), "required": ["volume"]},
    "alpha098": {"func": lambda dfs: alpha098(dfs["close"]), "required": ["close"]},
    "alpha099": {"func": lambda dfs: alpha099(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha100": {"func": lambda dfs: alpha100(dfs["volume"]), "required": ["volume"]},
    "alpha101": {"func": lambda dfs: alpha101(dfs["close"], dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["close", "high", "vwap", "volume"]},
    "alpha102": {"func": lambda dfs: alpha102(dfs["volume"]), "required": ["volume"]},
    "alpha103": {"func": lambda dfs: alpha103(dfs["low"]), "required": ["low"]},
    "alpha104": {"func": lambda dfs: alpha104(dfs["close"], dfs["high"], dfs["volume"]), "required": ["close", "high", "volume"]},
    "alpha105": {"func": lambda dfs: alpha105(dfs["open"], dfs["volume"]), "required": ["open", "volume"]},
    "alpha106": {"func": lambda dfs: alpha106(dfs["close"]), "required": ["close"]},
    "alpha107": {"func": lambda dfs: alpha107(dfs["open"], dfs["high"], dfs["low"], dfs["close"]), "required": ["open", "high", "low", "close"]},
    "alpha108": {"func": lambda dfs: alpha108(dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["high", "vwap", "volume"]},
    "alpha109": {"func": lambda dfs: alpha109(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha110": {"func": lambda dfs: alpha110(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha111": {"func": lambda dfs: alpha111(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha112": {"func": lambda dfs: alpha112(dfs["close"]), "required": ["close"]},
    "alpha113": {"func": lambda dfs: alpha113(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha114": {"func": lambda dfs: alpha114(dfs["close"], dfs["high"], dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["close", "high", "low", "vwap", "volume"]},
    "alpha115": {"func": lambda dfs: alpha115(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha116": {"func": lambda dfs: alpha116(dfs["close"]), "required": ["close"]},
    "alpha117": {"func": lambda dfs: alpha117(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha118": {"func": lambda dfs: alpha118(dfs["open"], dfs["high"], dfs["low"]), "required": ["open", "high", "low"]},
    "alpha119": {"func": lambda dfs: alpha119(dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["open", "vwap", "volume"]},
    "alpha120": {"func": lambda dfs: alpha120(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha121": {"func": lambda dfs: alpha121(dfs["vwap"], dfs["volume"]), "required": ["vwap", "volume"]},
    "alpha122": {"func": lambda dfs: alpha122(dfs["close"]), "required": ["close"]},
    "alpha123": {"func": lambda dfs: alpha123(dfs["high"], dfs["low"], dfs["volume"]), "required": ["high", "low", "volume"]},
    "alpha124": {"func": lambda dfs: alpha124(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha125": {"func": lambda dfs: alpha125(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha126": {"func": lambda dfs: alpha126(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha127": {"func": lambda dfs: alpha127(dfs["close"]), "required": ["close"]},
    "alpha128": {"func": lambda dfs: alpha128(dfs["high"], dfs["low"], dfs["close"], dfs["volume"]), "required": ["high", "low", "close", "volume"]},
    "alpha129": {"func": lambda dfs: alpha129(dfs["close"]), "required": ["close"]},
    "alpha130": {"func": lambda dfs: alpha130(dfs["high"], dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["high", "low", "vwap", "volume"]},
    "alpha131": {"func": lambda dfs: alpha131(dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["close", "vwap", "volume"]},
    "alpha132": {"func": lambda dfs: alpha132(dfs["amount"]), "required": ["amount"]},
    "alpha133": {"func": lambda dfs: alpha133(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha134": {"func": lambda dfs: alpha134(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha135": {"func": lambda dfs: alpha135(dfs["close"]), "required": ["close"]},
    "alpha136": {"func": lambda dfs: alpha136(dfs["open"], dfs["close"], dfs["volume"]), "required": ["open", "close", "volume"]},
    "alpha137": {"func": lambda dfs: alpha137(dfs["close"], dfs["open"], dfs["high"], dfs["low"]), "required": ["close", "open", "high", "low"]},
    "alpha138": {"func": lambda dfs: alpha138(dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["low", "vwap", "volume"]},
    "alpha139": {"func": lambda dfs: alpha139(dfs["open"], dfs["volume"]), "required": ["open", "volume"]},
    "alpha140": {"func": lambda dfs: alpha140(dfs["open"], dfs["high"], dfs["low"], dfs["close"], dfs["volume"]), "required": ["open", "high", "low", "close", "volume"]},
    "alpha141": {"func": lambda dfs: alpha141(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha142": {"func": lambda dfs: alpha142(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha143": {"func": lambda dfs: alpha143(dfs["close"]), "required": ["close"]},
    "alpha144": {"func": lambda dfs: alpha144(dfs["close"], dfs["amount"]), "required": ["close", "amount"]},
    "alpha145": {"func": lambda dfs: alpha145(dfs["volume"]), "required": ["volume"]},
    "alpha146": {"func": lambda dfs: alpha146(dfs["close"]), "required": ["close"]},
    "alpha147": {"func": lambda dfs: alpha147(dfs["close"]), "required": ["close"]},
    "alpha148": {"func": lambda dfs: alpha148(dfs["open"], dfs["volume"]), "required": ["open", "volume"]},
    "alpha149": {"func": lambda dfs: alpha149(dfs["benchmarkindex"], dfs["close"]), "required": ["close", "benchmarkindex"]},
    "alpha150": {"func": lambda dfs: alpha150(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha151": {"func": lambda dfs: alpha151(dfs["close"]), "required": ["close"]},
    "alpha152": {"func": lambda dfs: alpha152(dfs["close"]), "required": ["close"]},
    "alpha153": {"func": lambda dfs: alpha153(dfs["close"]), "required": ["close"]},
    "alpha154": {"func": lambda dfs: alpha154(dfs["vwap"], dfs["volume"]), "required": ["vwap", "volume"]},
    "alpha155": {"func": lambda dfs: alpha155(dfs["volume"]), "required": ["volume"]},
    "alpha156": {"func": lambda dfs: alpha156(dfs["open"], dfs["low"], dfs["vwap"]), "required": ["open", "low", "vwap"]},
    "alpha157": {"func": lambda dfs: alpha157(dfs["close"]), "required": ["close"]},
    "alpha158": {"func": lambda dfs: alpha158(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha159": {"func": lambda dfs: alpha159(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha160": {"func": lambda dfs: alpha160(dfs["close"]), "required": ["close"]},
    "alpha161": {"func": lambda dfs: alpha161(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha162": {"func": lambda dfs: alpha162(dfs["close"]), "required": ["close"]},
    "alpha163": {"func": lambda dfs: alpha163(dfs["close"], dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["close", "high", "vwap", "volume"]},
    "alpha164": {"func": lambda dfs: alpha164(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha165": {"func": lambda dfs: alpha165(dfs["close"]), "required": ["close"]},
    "alpha166": {"func": lambda dfs: alpha166(dfs["close"]), "required": ["close"]},
    "alpha167": {"func": lambda dfs: alpha167(dfs["close"]), "required": ["close"]},
    "alpha168": {"func": lambda dfs: alpha168(dfs["volume"]), "required": ["volume"]},
    "alpha169": {"func": lambda dfs: alpha169(dfs["close"]), "required": ["close"]},
    "alpha170": {"func": lambda dfs: alpha170(dfs["close"], dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["close", "high", "vwap", "volume"]},
    "alpha171": {"func": lambda dfs: alpha171(dfs["open"], dfs["close"], dfs["high"], dfs["low"]), "required": ["open", "close", "high", "low"]},
    "alpha172": {"func": lambda dfs: alpha172(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha173": {"func": lambda dfs: alpha173(dfs["close"]), "required": ["close"]},
    "alpha174": {"func": lambda dfs: alpha174(dfs["close"]), "required": ["close"]},
    "alpha175": {"func": lambda dfs: alpha175(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha176": {"func": lambda dfs: alpha176(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha177": {"func": lambda dfs: alpha177(dfs["high"]), "required": ["high"]},
    "alpha178": {"func": lambda dfs: alpha178(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha179": {"func": lambda dfs: alpha179(dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["low", "vwap", "volume"]},
    "alpha180": {"func": lambda dfs: alpha180(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha181": {"func": lambda dfs: alpha181(dfs["benchmarkindex"], dfs["close"]),"required": ["close", "benchmarkindex"]},
    "alpha182": {"func": lambda dfs: alpha182(dfs["benchmarkindex"], dfs["close"], dfs["open"]),"required": ["close", "open", "benchmarkindex"]},
    "alpha183": {"func": lambda dfs: alpha183(dfs["close"]), "required": ["close"]},
    "alpha184": {"func": lambda dfs: alpha184(dfs["open"], dfs["close"]), "required": ["open", "close"]},
    "alpha185": {"func": lambda dfs: alpha185(dfs["open"], dfs["close"]), "required": ["open", "close"]},
    "alpha186": {"func": lambda dfs: alpha186(dfs["close"], dfs["high"], dfs["low"]), "required": ["close", "high", "low"]},
    "alpha187": {"func": lambda dfs: alpha187(dfs["open"], dfs["high"], dfs["low"]), "required": ["open", "high", "low"]},
    "alpha188": {"func": lambda dfs: alpha188(dfs["high"], dfs["low"]), "required": ["high", "low"]},
    "alpha189": {"func": lambda dfs: alpha189(dfs["close"]), "required": ["close"]},
    "alpha190": {"func": lambda dfs: alpha190(dfs["close"]), "required": ["close"]},
    "alpha191": {"func": lambda dfs: alpha191(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
}


# ─────────────────────────────────────────────
# 通用算子（alpha021 之后开始用到，前20个未用到）
# ─────────────────────────────────────────────
def _decaylinear(df, window):
    """
    线性衰减加权：权重为 window, window-1, ..., 1（归一化），最近的观测权重最大
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


def _wma(df, window):
    """
    国君报告里的 WMA：权重按 0.9^0, 0.9^1, ..., 0.9^(window-1) 由近到远指数衰减（归一化）
    区别于 DECAYLINEAR 的线性衰减
    """
    weights = (0.9 ** np.arange(window))[::-1]  # 反转，使窗口末尾（最近的观测）对应 0.9^0
    weights = weights / weights.sum()
    return df.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


def build_ff3_factors(close, marketcap, valuation, index_close,
                      universe_snapshots, valuation_type="PE",
                      start=None, end=None, return_details=False):
    """
    为 Alpha#30 构造日频 MKT/SMB/HML 三因子。

    构造口径
    --------
    1. 股票池：每个形成日使用 universe_snapshots 中不晚于该日的最近一次
       沪深300+中证500成分股快照；绝不使用未来快照兜底。
    2. 调仓：每月最后一个交易日独立做 2×2 排序，分组从下一交易日起生效，
       持有至下一个月末。
    3. 规模：按形成日总市值的截面中位数分为 Small / Big。
    4. 价值：PE 或 PB 先转成 E/P 或 B/M（即取倒数），再按截面中位数
       分为 High / Low；PE/PB 非正数及缺失值不参与分组。
    5. 组合收益：SH、SL、BH、BL 四组合均使用前一交易日总市值加权。
    6. MKT：由传入的中证800指数收盘价计算普通日收益率，不减无风险利率。

    参数
    ----
    close : DataFrame
        前复权收盘价，index 为交易日，columns 为股票代码。
    marketcap : DataFrame
        每日总市值，结构与 close 相同。
    valuation : DataFrame
        每日 PE 或 PB，结构与 close 相同。由 valuation_type 指明类型。
    index_close : Series 或 DataFrame
        中证800指数收盘价。若为 DataFrame，必须包含 ``close`` 列；
        只有一列时也可直接传入。
    universe_snapshots : dict
        ``{快照日期: 股票代码集合}``。快照日期必须是当时已经可得的历史快照。
    valuation_type : {"PE", "PB"}, default "PE"
        valuation 的口径。PE 对应 E/P 价值因子，PB 对应 B/M 价值因子。
    start, end : str 或 Timestamp, optional
        最终输出区间。为保证首月能够取得分组，内部仍会使用 start 之前的数据。
    return_details : bool, default False
        False 只返回三因子 DataFrame；True 额外返回四组合收益和月末分组记录。

    返回
    ----
    factors : DataFrame
        index 为交易日，columns 为 ``mkt``, ``smb``, ``hml``。
    portfolios : DataFrame, optional
        return_details=True 时返回，columns 为 ``SH``, ``SL``, ``BH``, ``BL``。
    membership : DataFrame, optional
        return_details=True 时返回，记录每个形成日各股票所属组合。
    """
    if not isinstance(close, pd.DataFrame):
        raise TypeError("close 必须是 DataFrame")
    if not isinstance(marketcap, pd.DataFrame):
        raise TypeError("marketcap 必须是 DataFrame")
    if not isinstance(valuation, pd.DataFrame):
        raise TypeError("valuation 必须是 DataFrame")
    if not universe_snapshots:
        raise ValueError("universe_snapshots 不能为空")

    valuation_type = str(valuation_type).upper()
    if valuation_type not in {"PE", "PB"}:
        raise ValueError("valuation_type 只能是 'PE' 或 'PB'")

    # 统一股票和日期，并保证时间升序、无重复。
    common_dates = close.index.intersection(marketcap.index).intersection(valuation.index)
    common_stocks = close.columns.intersection(marketcap.columns).intersection(valuation.columns)
    common_dates = pd.DatetimeIndex(pd.to_datetime(common_dates)).drop_duplicates().sort_values()
    common_stocks = pd.Index(common_stocks).drop_duplicates()

    if len(common_dates) < 2 or len(common_stocks) == 0:
        raise ValueError("close、marketcap、valuation 没有足够的公共日期或股票")

    close = close.reindex(index=common_dates, columns=common_stocks).astype(float)
    marketcap = marketcap.reindex(index=common_dates, columns=common_stocks).astype(float)
    valuation = valuation.reindex(index=common_dates, columns=common_stocks).astype(float)

    if isinstance(index_close, pd.DataFrame):
        if "close" in index_close.columns:
            index_close = index_close["close"]
        elif index_close.shape[1] == 1:
            index_close = index_close.iloc[:, 0]
        else:
            raise ValueError("index_close 为 DataFrame 时必须包含 close 列或只有一列")
    if not isinstance(index_close, pd.Series):
        raise TypeError("index_close 必须是 Series 或 DataFrame")

    index_close = index_close.copy()
    index_close.index = pd.to_datetime(index_close.index)
    index_close = index_close[~index_close.index.duplicated(keep="last")].sort_index()
    index_close = pd.to_numeric(index_close, errors="coerce").reindex(common_dates)

    snapshots = {
        pd.Timestamp(snapshot_date): set(codes)
        for snapshot_date, codes in universe_snapshots.items()
        if codes
    }
    snapshot_dates = sorted(snapshots)
    if not snapshot_dates:
        raise ValueError("universe_snapshots 中没有有效快照")

    # 月末交易日既是本月形成日，也是上月组合的最后持有日。
    date_series = pd.Series(common_dates, index=common_dates)
    formation_dates = pd.DatetimeIndex(
        date_series.groupby(date_series.index.to_period("M")).last().values
    )

    stock_ret = close.pct_change(fill_method=None)
    lagged_cap = marketcap.shift(1)
    value_score = 1.0 / valuation.where(valuation > 0)

    portfolios = pd.DataFrame(
        np.nan,
        index=common_dates,
        columns=["SH", "SL", "BH", "BL"],
        dtype=float,
    )
    membership_records = []

    def weighted_return(date, stocks):
        stocks = pd.Index(stocks).intersection(common_stocks)
        if len(stocks) == 0:
            return np.nan

        returns = stock_ret.loc[date, stocks]
        weights = lagged_cap.loc[date, stocks]
        valid = returns.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return np.nan

        returns = returns[valid]
        weights = weights[valid]
        return np.average(returns.to_numpy(), weights=weights.to_numpy())

    for i, formation_date in enumerate(formation_dates[:-1]):
        eligible_snapshots = [d for d in snapshot_dates if d <= formation_date]
        if not eligible_snapshots:
            # 不使用未来股票池，避免首段样本产生前视偏差。
            continue

        universe = pd.Index(snapshots[max(eligible_snapshots)]).intersection(common_stocks)
        if len(universe) == 0:
            continue

        cross_section = pd.DataFrame({
            "marketcap": marketcap.loc[formation_date, universe],
            "value": value_score.loc[formation_date, universe],
            "close": close.loc[formation_date, universe],
        }).replace([np.inf, -np.inf], np.nan).dropna()
        cross_section = cross_section[
            (cross_section["marketcap"] > 0) &
            (cross_section["close"] > 0)
        ]
        if len(cross_section) < 20:
            continue

        size_median = cross_section["marketcap"].median()
        value_median = cross_section["value"].median()

        cross_section["size_group"] = np.where(
            cross_section["marketcap"] <= size_median, "S", "B"
        )
        cross_section["value_group"] = np.where(
            cross_section["value"] > value_median, "H", "L"
        )
        cross_section["portfolio"] = (
            cross_section["size_group"] + cross_section["value_group"]
        )

        for stock, row in cross_section.iterrows():
            membership_records.append({
                "formation_date": formation_date,
                "ts_code": stock,
                "size_group": row["size_group"],
                "value_group": row["value_group"],
                "portfolio": row["portfolio"],
            })

        next_formation_date = formation_dates[i + 1]
        holding_dates = common_dates[
            (common_dates > formation_date) &
            (common_dates <= next_formation_date)
        ]

        members = {
            name: cross_section.index[cross_section["portfolio"] == name]
            for name in portfolios.columns
        }
        for date in holding_dates:
            for name, stocks in members.items():
                portfolios.loc[date, name] = weighted_return(date, stocks)

    factors = pd.DataFrame(index=common_dates, dtype=float)
    factors["mkt"] = index_close.pct_change(fill_method=None)
    factors["smb"] = (
        portfolios[["SH", "SL"]].mean(axis=1, skipna=False) -
        portfolios[["BH", "BL"]].mean(axis=1, skipna=False)
    )
    factors["hml"] = (
        portfolios[["SH", "BH"]].mean(axis=1, skipna=False) -
        portfolios[["SL", "BL"]].mean(axis=1, skipna=False)
    )

    output_start = common_dates.min() if start is None else pd.Timestamp(start)
    output_end = common_dates.max() if end is None else pd.Timestamp(end)
    factors = factors.loc[
        (factors.index >= output_start) & (factors.index <= output_end)
    ]
    portfolios = portfolios.reindex(factors.index)

    if not return_details:
        return factors

    membership = pd.DataFrame(
        membership_records,
        columns=[
            "formation_date", "ts_code", "size_group",
            "value_group", "portfolio",
        ],
    )
    return factors, portfolios, membership


def _regbeta_seq(df, window):
    """
    滚动窗口内，把 df 的值当作 y，与时间序列 x=[1..window] 做线性回归，返回斜率(beta)
    """
    x = np.arange(1, window + 1, dtype=float)
    x_demeaned = x - x.mean()
    denom = np.sum(x_demeaned ** 2)

    def slope(y):
        if np.isnan(y).any():
            return np.nan
        return np.dot(x_demeaned, y - y.mean()) / denom

    return df.rolling(window).apply(slope, raw=True)


def _lowday(df, window):
    """
    LOWDAY(X,N)：过去N日窗口内最小值出现距今的天数（当天为0，最早一天为 N-1）
    """
    def argmin_dist(x):
        if np.isnan(x).any():
            return np.nan
        idx = np.argmin(x)  # 0=窗口最早一天，len-1=当天
        return (len(x) - 1) - idx

    return df.rolling(window).apply(argmin_dist, raw=True)


def _highday(df, window):
    """
    HIGHDAY(X,N)：过去N日窗口内最大值出现距今的天数（当天为0，最早一天为 N-1）
    """
    def argmax_dist(x):
        if np.isnan(x).any():
            return np.nan
        idx = np.argmax(x)
        return (len(x) - 1) - idx

    return df.rolling(window).apply(argmax_dist, raw=True)


def alpha001(close, open, volume):
    """
    Alpha#01: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    """
    delta_log_vol = np.log(volume).diff(1)
    ret_co = (close - open) / open

    rank_delta = delta_log_vol.rank(axis=1, pct=True)
    rank_ret = ret_co.rank(axis=1, pct=True)

    alpha = -1 * rank_delta.rolling(6).corr(rank_ret)
    return alpha


def alpha002(close, high, low):
    """
    Alpha#02: (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    """
    inner = ((close - low) - (high - close)) / (high - low)
    alpha = -1 * inner.diff(1)
    return alpha


def alpha003(close, high, low):
    """
    Alpha#03: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    """
    delay_close = close.shift(1)

    cond_eq = close == delay_close
    cond_up = close > delay_close

    term_up = close - np.minimum(low, delay_close)
    term_down = close - np.maximum(high, delay_close)

    inner = pd.DataFrame(
        np.where(cond_eq, 0.0, np.where(cond_up, term_up, term_down)),
        index=close.index,
        columns=close.columns
    )
    alpha = inner.rolling(6).sum()
    return alpha


def alpha004(close, volume):
    """
    Alpha#04:
    if (mean(close,8)+std(close,8)) < mean(close,2): -1
    elif mean(close,2) < (mean(close,8)-std(close,8)): 1
    elif volume/mean(volume,20) >= 1: 1
    else: -1
    """
    mean8 = close.rolling(8).mean()
    std8 = close.rolling(8).std()
    mean2 = close.rolling(2).mean()
    vol_ratio = volume / volume.rolling(20).mean()

    cond1 = (mean8 + std8) < mean2
    cond2 = mean2 < (mean8 - std8)
    cond3 = vol_ratio >= 1

    # 优先级 cond1 > cond2 > cond3 > 默认-1，因此按相反顺序赋值，让高优先级最后覆盖
    alpha = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    alpha[cond3] = 1.0
    alpha[cond2] = 1.0
    alpha[cond1] = -1.0
    return alpha


def alpha005(high, volume):
    """
    Alpha#05: (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    """
    tsrank_vol = volume.rolling(5).rank(pct=True)
    tsrank_high = high.rolling(5).rank(pct=True)
    corr = tsrank_vol.rolling(5).corr(tsrank_high)
    alpha = -1 * corr.rolling(3).max()
    return alpha


def alpha006(open, high):
    """
    Alpha#06: (RANK(SIGN(DELTA(((OPEN * 0.85) + (HIGH * 0.15)), 4))) * -1)
    """
    inner = (open * 0.85) + (high * 0.15)
    delta4 = inner.diff(4)
    alpha = -1 * np.sign(delta4).rank(axis=1, pct=True)
    return alpha


def alpha007(close, vwap, volume):
    """
    Alpha#07: ((RANK(TSMAX((VWAP - CLOSE), 3)) + RANK(TSMIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    """
    diff = vwap - close
    part1 = diff.rolling(3).max().rank(axis=1, pct=True)
    part2 = diff.rolling(3).min().rank(axis=1, pct=True)
    part3 = volume.diff(3).rank(axis=1, pct=True)
    alpha = (part1 + part2) * part3
    return alpha


def alpha008(high, low, vwap):
    """
    Alpha#08: RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    """
    inner = (((high + low) / 2) * 0.2) + (vwap * 0.8)
    delta4 = inner.diff(4)
    alpha = (-1 * delta4).rank(axis=1, pct=True)
    return alpha


def alpha009(high, low, volume):
    """
    Alpha#09: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)
    SMA(X,N,M) 递归定义 Y_t = (M*X_t + (N-M)*Y_{t-1}) / N，等价于 alpha=M/N 的 EWM（adjust=False）
    """
    mid = (high + low) / 2
    delay_mid = (high.shift(1) + low.shift(1)) / 2
    inner = (mid - delay_mid) * (high - low) / volume
    alpha = inner.ewm(alpha=2 / 7, adjust=False).mean()
    return alpha


def alpha010(close):
    """
    Alpha#10: RANK(TSMAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2, 5))
    """
    ret = close.pct_change()
    std20 = ret.rolling(20).std()
    inner = pd.DataFrame(
        np.where(ret < 0, std20, close),
        index=close.index,
        columns=close.columns
    )
    squared = inner ** 2
    alpha = squared.rolling(5).max().rank(axis=1, pct=True)
    return alpha


def alpha011(close, high, low, volume):
    """
    Alpha#11: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 6)
    """
    inner = ((close - low) - (high - close)) / (high - low) * volume
    alpha = inner.rolling(6).sum()
    return alpha


def alpha012(open, close, vwap):
    """
    Alpha#12: (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    """
    part1 = (open - vwap.rolling(10).mean()).rank(axis=1, pct=True)
    part2 = -1 * (close - vwap).abs().rank(axis=1, pct=True)
    alpha = part1 * part2
    return alpha


def alpha013(high, low, vwap):
    """
    Alpha#13: (((HIGH * LOW)^0.5) - VWAP)
    """
    alpha = (high * low) ** 0.5 - vwap
    return alpha


def alpha014(close):
    """
    Alpha#14: CLOSE - DELAY(CLOSE, 5)
    """
    alpha = close - close.shift(5)
    return alpha


def alpha015(open, close):
    """
    Alpha#15: OPEN / DELAY(CLOSE, 1) - 1
    """
    alpha = open / close.shift(1) - 1
    return alpha


def alpha016(volume, vwap):
    """
    Alpha#16: (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    """
    rank_vol = volume.rank(axis=1, pct=True)
    rank_vwap = vwap.rank(axis=1, pct=True)
    corr = rank_vol.rolling(5).corr(rank_vwap)
    rank_corr = corr.rank(axis=1, pct=True)
    alpha = -1 * rank_corr.rolling(5).max()
    return alpha


def alpha017(close, vwap):
    """
    Alpha#17: RANK((VWAP - TSMAX(VWAP, 15))) ^ DELTA(CLOSE, 5)
    先做截面rank把底数收敛到[0,1]，避免负数底数配分数次幂产生复数
    """
    base = (vwap - vwap.rolling(15).max()).rank(axis=1, pct=True)
    exponent = close.diff(5)
    alpha = base ** exponent
    return alpha


def alpha018(close):
    """
    Alpha#18: CLOSE / DELAY(CLOSE, 5)
    """
    alpha = close / close.shift(5)
    return alpha


def alpha019(close):
    """
    Alpha#19:
    if close < delay(close,5): (close-delay(close,5))/delay(close,5)
    elif close == delay(close,5): 0
    else: (close-delay(close,5))/close
    """
    delay5 = close.shift(5)
    cond_down = close < delay5
    cond_eq = close == delay5

    down_val = (close - delay5) / delay5
    up_val = (close - delay5) / close

    alpha = pd.DataFrame(
        np.where(cond_down, down_val, np.where(cond_eq, 0.0, up_val)),
        index=close.index,
        columns=close.columns
    )
    return alpha


def alpha020(close):
    """
    Alpha#20: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100
    """
    alpha = (close - close.shift(6)) / close.shift(6) * 100
    return alpha


def alpha021(close):
    """
    Alpha#21: REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
    过去6日 MEAN(CLOSE,6) 序列对时间序号[1..6]做回归，取斜率
    """
    mean6 = close.rolling(6).mean()
    alpha = _regbeta_seq(mean6, 6)
    return alpha


def alpha022(close):
    """
    Alpha#22: SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    原文写的是 SMEAN，按 SMA(N=12,M=1) 处理
    """
    mean6 = close.rolling(6).mean()
    ratio = (close - mean6) / mean6
    inner = ratio - ratio.shift(3)
    alpha = inner.ewm(alpha=1 / 12, adjust=False).mean()
    return alpha


def alpha023(close):
    """
    Alpha#23: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) /
              (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    """
    std20 = close.rolling(20).std()
    delay_close = close.shift(1)
    up = pd.DataFrame(np.where(close > delay_close, std20, 0.0), index=close.index, columns=close.columns)
    down = pd.DataFrame(np.where(close <= delay_close, std20, 0.0), index=close.index, columns=close.columns)
    sma_up = up.ewm(alpha=1 / 20, adjust=False).mean()
    sma_down = down.ewm(alpha=1 / 20, adjust=False).mean()
    alpha = sma_up / (sma_up + sma_down) * 100
    return alpha


def alpha024(close):
    """
    Alpha#24: SMA(CLOSE-DELAY(CLOSE,5),5,1)
    """
    alpha = (close - close.shift(5)).ewm(alpha=1 / 5, adjust=False).mean()
    return alpha


def alpha025(close, volume):
    """
    Alpha#25: ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250))))
    """
    ret = close.pct_change()
    delta7 = close.diff(7)
    vol_ratio = volume / volume.rolling(20).mean()
    decay9 = _decaylinear(vol_ratio, 9)

    part1 = -1 * (delta7 * (1 - decay9.rank(axis=1, pct=True))).rank(axis=1, pct=True)
    part2 = 1 + ret.rolling(250).sum().rank(axis=1, pct=True)
    alpha = part1 * part2
    return alpha


def alpha026(close, vwap):
    """
    Alpha#26: (((SUM(CLOSE,7)/7)-CLOSE)) + (CORR(VWAP,DELAY(CLOSE,5),230))
    """
    part1 = close.rolling(7).sum() / 7 - close
    part2 = vwap.rolling(230).corr(close.shift(5))
    alpha = part1 + part2
    return alpha


def alpha027(close):
    """
    Alpha#27: WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    WMA为0.9指数衰减加权（区别于DECAYLINEAR的线性衰减），见 _wma
    """
    part1 = (close - close.shift(3)) / close.shift(3) * 100
    part2 = (close - close.shift(6)) / close.shift(6) * 100
    inner = part1 + part2
    alpha = _wma(inner, 12)
    return alpha


def alpha028(close, high, low):
    """
    Alpha#28: 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
              -2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    按原文字面实现：两项分子相同（CLOSE-TSMIN(LOW,9)），分母不同——
    第一项分母 TSMAX(HIGH,9)-TSMIN(LOW,9)；第二项分母 MAX(HIGH,9)-TSMAX(LOW,9)，
    这里的 MAX(HIGH,9) 按本文档一贯约定视为滚动窗口 TSMAX(HIGH,9)。
    """
    tsmin_low9 = low.rolling(9).min()
    tsmax_low9 = low.rolling(9).max()
    tsmax_high9 = high.rolling(9).max()

    rsv1 = (close - tsmin_low9) / (tsmax_high9 - tsmin_low9) * 100
    term1 = 3 * rsv1.ewm(alpha=1 / 3, adjust=False).mean()

    rsv2 = (close - tsmin_low9) / (tsmax_high9 - tsmax_low9) * 100
    k2 = rsv2.ewm(alpha=1 / 3, adjust=False).mean()
    d2 = k2.ewm(alpha=1 / 3, adjust=False).mean()
    term2 = 2 * d2

    alpha = term1 - term2
    return alpha


def alpha029(close, volume):
    """
    Alpha#29: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    """
    alpha = (close - close.shift(6)) / close.shift(6) * volume
    return alpha


def alpha030(dfs, regression_window=60, wma_window=20):
    """
    Alpha#30: WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

    计算时先调用 build_ff3_factors 构造 MKT、SMB、HML，
    再进行60日滚动三因子回归并计算20日WMA。

    对每只股票的普通日收益率做带截距的三因子时间序列回归：
        RET_i = intercept + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + residual

    每个交易日使用截至当日的最近 regression_window 个完整观测拟合回归，
    REGRESI 取窗口最后一天（当日）的回归残差；残差平方后使用国君口径
    的指数衰减 WMA（_wma）计算最终因子值。

    参数
    ----
    dfs : dict
        包含 close、cap、pe、benchmarkindex 和 universe_snapshots。
    regression_window : int, default 60
        滚动回归窗口。
    wma_window : int, default 20
        残差平方的 WMA 窗口。

    返回
    ----
    DataFrame
        与 dfs["close"] 具有相同 index/columns 的 Alpha30 面板。
    """
    close = dfs["close"]
    ff3 = build_ff3_factors(close=close,marketcap=dfs["cap"],valuation=dfs["pe"],index_close=dfs["benchmarkindex"]["close"],universe_snapshots=dfs["universe_snapshots"],valuation_type="PE",)
    mkt = ff3["mkt"]
    smb = ff3["smb"]
    hml = ff3["hml"]

    factor_df = pd.concat([mkt.rename("mkt"), smb.rename("smb"), hml.rename("hml")],axis=1,).reindex(close.index)
    stock_ret = close.pct_change(fill_method=None)

    # X 包含截距项。因子对所有股票共享，因此每个交易日只需求解一次
    # 4×4 的 X'X，再同时处理当天所有满足完整窗口要求的股票。
    x = pd.DataFrame({"intercept": 1.0,"mkt": factor_df["mkt"],"smb": factor_df["smb"],"hml": factor_df["hml"],}, index=close.index)
    x_columns = list(x.columns)
    k = len(x_columns)

    factor_window_valid = (factor_df.notna().all(axis=1).rolling(regression_window).sum().eq(regression_window))
    stock_window_valid = (stock_ret.notna().rolling(regression_window).sum().eq(regression_window))

    # 逐日滚动生成 X'X。
    xtx = np.full((len(x), k, k), np.nan, dtype=float)
    for row_i, col_i in enumerate(x_columns):
        for row_j, col_j in enumerate(x_columns):
            xtx[:, row_i, row_j] = ((x[col_i] * x[col_j]).rolling(regression_window).sum().to_numpy())

    # 对所有股票一次生成各分量的滚动 X'Y。
    xty = np.full((len(x), k, len(stock_ret.columns)),np.nan,dtype=float,)
    for row_i, col_i in enumerate(x_columns):
        xty[:, row_i, :] = (stock_ret.mul(x[col_i], axis=0).rolling(regression_window).sum().to_numpy())

    residual = pd.DataFrame(np.nan,index=close.index,columns=close.columns,dtype=float,)
    stock_ret_values = stock_ret.to_numpy()
    x_values = x.to_numpy()
    stock_valid_values = stock_window_valid.to_numpy()
    factor_valid_values = factor_window_valid.to_numpy()

    for date_pos in range(regression_window - 1, len(close.index)):
        if not factor_valid_values[date_pos]:
            continue

        valid_stocks = (
            stock_valid_values[date_pos] &
            np.isfinite(stock_ret_values[date_pos])
        )
        if not valid_stocks.any():
            continue

        matrix = xtx[date_pos]
        # 分两步索引，避免 NumPy 高级索引把股票轴移动到最前面。
        rhs = xty[date_pos][:, valid_stocks]
        if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
            continue

        try:
            beta = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            # 因子在窗口内完全共线时不强行使用伪逆，避免产生不稳定结果。
            continue

        fitted_today = x_values[date_pos] @ beta
        residual.iloc[date_pos, np.flatnonzero(valid_stocks)] = (
            stock_ret_values[date_pos, valid_stocks] - fitted_today
        )

    return _wma(residual.pow(2), wma_window)


def alpha031(close):
    """
    Alpha#31: (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    """
    mean12 = close.rolling(12).mean()
    alpha = (close - mean12) / mean12 * 100
    return alpha


def alpha032(high, volume):
    """
    Alpha#32: (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))
    """
    rank_high = high.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr = rank_high.rolling(3).corr(rank_vol)
    alpha = -1 * corr.rank(axis=1, pct=True).rolling(3).sum()
    return alpha


def alpha033(close, low, volume):
    """
    Alpha#33: ((((-1*TSMIN(LOW,5))+DELAY(TSMIN(LOW,5),5))*RANK(((SUM(RET,240)-SUM(RET,20))/220)))*TSRANK(VOLUME,5))
    """
    ret = close.pct_change()
    tsmin_low5 = low.rolling(5).min()
    part1 = -1 * tsmin_low5 + tsmin_low5.shift(5)
    part2 = ((ret.rolling(240).sum() - ret.rolling(20).sum()) / 220).rank(axis=1, pct=True)
    part3 = volume.rolling(5).rank(pct=True)
    alpha = part1 * part2 * part3
    return alpha


def alpha034(close):
    """
    Alpha#34: MEAN(CLOSE,12)/CLOSE
    """
    alpha = close.rolling(12).mean() / close
    return alpha


def alpha035(open, volume):
    """
    Alpha#35: (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),
                   RANK(DECAYLINEAR(CORR(VOLUME,((OPEN*0.65)+(OPEN*0.35)),17),7))) * -1)
    (OPEN*0.65)+(OPEN*0.35) 字面化简后等于 OPEN，按原始公式字面实现
    """
    part1 = _decaylinear(open.diff(1), 15).rank(axis=1, pct=True)
    weighted_open = (open * 0.65) + (open * 0.35)
    corr = volume.rolling(17).corr(weighted_open)
    part2 = _decaylinear(corr, 7).rank(axis=1, pct=True)
    alpha = -1 * np.minimum(part1, part2)
    return alpha


def alpha036(volume, vwap):
    """
    Alpha#36: RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2))
    原文括号有缺失（"SUM(CORR(...),6),2)"），按最合理的读法处理：
    CORR窗口=6，SUM窗口=2，最外层RANK
    """
    rank_vol = volume.rank(axis=1, pct=True)
    rank_vwap = vwap.rank(axis=1, pct=True)
    corr = rank_vol.rolling(6).corr(rank_vwap)
    alpha = corr.rolling(2).sum().rank(axis=1, pct=True)
    return alpha


def alpha037(open, close):
    """
    Alpha#37: (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
    """
    ret = close.pct_change()
    inner = open.rolling(5).sum() * ret.rolling(5).sum()
    alpha = -1 * (inner - inner.shift(10)).rank(axis=1, pct=True)
    return alpha


def alpha038(high):
    """
    Alpha#38: (((SUM(HIGH,20)/20)<HIGH) ? (-1*DELTA(HIGH,2)) : 0)
    """
    cond = (high.rolling(20).sum() / 20) < high
    alpha = pd.DataFrame(np.where(cond, -1 * high.diff(2), 0.0), index=high.index, columns=high.columns)
    return alpha


def alpha039(close, open, vwap, volume):
    """
    Alpha#39: ((RANK(DECAYLINEAR(DELTA(CLOSE,2),8)) -
                RANK(DECAYLINEAR(CORR(((VWAP*0.3)+(OPEN*0.7)),SUM(MEAN(VOLUME,180),37),14),12))) * -1)
    """
    part1 = _decaylinear(close.diff(2), 8).rank(axis=1, pct=True)
    weighted_price = (vwap * 0.3) + (open * 0.7)
    mean_vol_sum = volume.rolling(180).mean().rolling(37).sum()
    corr = weighted_price.rolling(14).corr(mean_vol_sum)
    part2 = _decaylinear(corr, 12).rank(axis=1, pct=True)
    alpha = -1 * (part1 - part2)
    return alpha


def alpha040(close, volume):
    """
    Alpha#40: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
    """
    delay_close = close.shift(1)
    up_vol = pd.DataFrame(np.where(close > delay_close, volume, 0.0), index=close.index, columns=close.columns)
    down_vol = pd.DataFrame(np.where(close <= delay_close, volume, 0.0), index=close.index, columns=close.columns)
    alpha = up_vol.rolling(26).sum() / down_vol.rolling(26).sum() * 100
    return alpha


def alpha041(vwap):
    """
    Alpha#41: (RANK(TSMAX(DELTA(VWAP,3),5))*-1)
    """
    alpha = -1 * vwap.diff(3).rolling(5).max().rank(axis=1, pct=True)
    return alpha


def alpha042(high, volume):
    """
    Alpha#42: ((-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10))
    """
    part1 = -1 * high.rolling(10).std().rank(axis=1, pct=True)
    part2 = high.rolling(10).corr(volume)
    alpha = part1 * part2
    return alpha


def alpha043(close, volume):
    """
    Alpha#43: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    """
    delay_close = close.shift(1)
    signed_vol = pd.DataFrame(
        np.where(close > delay_close, volume, np.where(close < delay_close, -volume, 0.0)),
        index=close.index, columns=close.columns
    )
    alpha = signed_vol.rolling(6).sum()
    return alpha


def alpha044(low, volume, vwap):
    """
    Alpha#44: (TSRANK(DECAYLINEAR(CORR(LOW,MEAN(VOLUME,10),7),6),4) + TSRANK(DECAYLINEAR(DELTA(VWAP,3),10),15))
    """
    corr = low.rolling(7).corr(volume.rolling(10).mean())
    part1 = _decaylinear(corr, 6).rolling(4).rank(pct=True)
    part2 = _decaylinear(vwap.diff(3), 10).rolling(15).rank(pct=True)
    alpha = part1 + part2
    return alpha


def alpha045(close, open, vwap, volume):
    """
    Alpha#45: (RANK(DELTA(((CLOSE*0.6)+(OPEN*0.4)),1)) * RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
    """
    weighted_price = (close * 0.6) + (open * 0.4)
    part1 = weighted_price.diff(1).rank(axis=1, pct=True)
    part2 = vwap.rolling(15).corr(volume.rolling(150).mean()).rank(axis=1, pct=True)
    alpha = part1 * part2
    return alpha


def alpha046(close):
    """
    Alpha#46: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    """
    alpha = (close.rolling(3).mean() + close.rolling(6).mean()
             + close.rolling(12).mean() + close.rolling(24).mean()) / (4 * close)
    return alpha


def alpha047(close, high, low):
    """
    Alpha#47: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    """
    tsmax_high6 = high.rolling(6).max()
    tsmin_low6 = low.rolling(6).min()
    inner = (tsmax_high6 - close) / (tsmax_high6 - tsmin_low6) * 100
    alpha = inner.ewm(alpha=1 / 9, adjust=False).mean()
    return alpha


def alpha048(close, volume):
    """
    Alpha#48: (-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))
               +SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
    """
    sign_sum = (np.sign(close - close.shift(1))
                + np.sign(close.shift(1) - close.shift(2))
                + np.sign(close.shift(2) - close.shift(3)))
    part1 = -1 * sign_sum.rank(axis=1, pct=True)
    alpha = part1 * volume.rolling(5).sum() / volume.rolling(20).sum()
    return alpha


def _dtm_dbm_sum_hl(high, low, window=12):
    """
    alpha49/50/51 共用：基于 HIGH+LOW 与前一日 HIGH+LOW 比较的 up/down move 量级，
    再做 window 期滚动求和（这套定义和 alpha069 里的 DTM/DBM 不是同一套，见 alpha069 注释）
    """
    delay_high = high.shift(1)
    delay_low = low.shift(1)
    term = np.maximum((high - delay_high).abs(), (low - delay_low).abs())
    sum_now = high + low
    sum_delay = delay_high + delay_low

    up_term = pd.DataFrame(np.where(sum_now <= sum_delay, 0.0, term), index=high.index, columns=high.columns)
    down_term = pd.DataFrame(np.where(sum_now >= sum_delay, 0.0, term), index=high.index, columns=high.columns)
    return up_term.rolling(window).sum(), down_term.rolling(window).sum()


def alpha049(high, low):
    """
    Alpha#49: SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
              / (同上 + SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(...)),12))
    """
    up_sum, down_sum = _dtm_dbm_sum_hl(high, low, 12)
    alpha = down_sum / (down_sum + up_sum)
    return alpha


def alpha050(high, low):
    """
    Alpha#50: SUM(down,12)/(SUM(down,12)+SUM(up,12)) - SUM(up,12)/(SUM(up,12)+SUM(down,12))
    """
    up_sum, down_sum = _dtm_dbm_sum_hl(high, low, 12)
    alpha = up_sum / (up_sum + down_sum) - down_sum / (down_sum + up_sum)
    return alpha


def alpha051(high, low):
    """
    Alpha#51: SUM(down,12)/(SUM(down,12)+SUM(up,12))
    """
    up_sum, down_sum = _dtm_dbm_sum_hl(high, low, 12)
    alpha = up_sum / (up_sum + down_sum)
    return alpha


def alpha052(high, low, close):
    """
    Alpha#52: SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
    原文用 "L" 代指 LOW
    """
    typical = (high + low + close) / 3
    delay_typical = typical.shift(1)
    up = (high - delay_typical).clip(lower=0)
    down = (delay_typical - low).clip(lower=0)
    alpha = up.rolling(26).sum() / down.rolling(26).sum() * 100
    return alpha


def alpha053(close):
    """
    Alpha#53: COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    """
    up = (close > close.shift(1)).astype(float)
    alpha = up.rolling(12).sum() / 12 * 100
    return alpha


def alpha054(close, open):
    """
    Alpha#54: (-1*RANK((STD(ABS(CLOSE-OPEN))+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
    原文 STD(ABS(CLOSE-OPEN)) 未给窗口，按和 CORR 一致取 10
    """
    part1 = (close - open).abs().rolling(10).std() + (close - open)
    part2 = close.rolling(10).corr(open)
    alpha = -1 * (part1 + part2).rank(axis=1, pct=True)
    return alpha


def alpha055(close, open, high, low):
    """
    Alpha#55: SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/DENOM
                  *MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))), 20)
    DENOM 是三支条件分支（类似 Welles Wilder ASI 摆动指标的真实波幅项），见分支定义
    """
    delay_close = close.shift(1)
    delay_open = open.shift(1)
    delay_low = low.shift(1)

    term_a = (high - delay_close).abs()
    term_b = (low - delay_close).abs()
    term_c = (high - delay_low).abs()
    term_d = (delay_close - delay_open).abs()

    cond1 = (term_a > term_b) & (term_a > term_c)
    cond2 = (term_b > term_c) & (term_b > term_a)

    denom1 = term_a + term_b / 2 + term_d / 4
    denom2 = term_b + term_a / 2 + term_d / 4
    denom3 = term_c + term_d / 4

    denom = pd.DataFrame(
        np.where(cond1, denom1, np.where(cond2, denom2, denom3)),
        index=close.index, columns=close.columns
    )

    numerator = 16 * (close - delay_close + (close - open) / 2 + delay_close - delay_open)
    max_term = np.maximum(term_a, term_b)

    inner = numerator / denom * max_term
    alpha = inner.rolling(20).sum()
    return alpha


def alpha056(open, high, low, volume):
    """
    Alpha#56: (RANK((OPEN-TSMIN(OPEN,12))) < RANK((RANK(CORR(SUM((HIGH+LOW)/2,19),SUM(MEAN(VOLUME,40),19),13))^5)))
    比较结果转成 1.0/0.0
    """
    part1 = (open - open.rolling(12).min()).rank(axis=1, pct=True)
    sum_mid = ((high + low) / 2).rolling(19).sum()
    sum_vol = volume.rolling(40).mean().rolling(19).sum()
    corr = sum_mid.rolling(13).corr(sum_vol)
    part2 = (corr.rank(axis=1, pct=True) ** 5).rank(axis=1, pct=True)
    alpha = (part1 < part2).astype(float)
    return alpha


def alpha057(close, high, low):
    """
    Alpha#57: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    """
    tsmin_low9 = low.rolling(9).min()
    tsmax_high9 = high.rolling(9).max()
    rsv = (close - tsmin_low9) / (tsmax_high9 - tsmin_low9) * 100
    alpha = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    return alpha


def alpha058(close):
    """
    Alpha#58: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    """
    up = (close > close.shift(1)).astype(float)
    alpha = up.rolling(20).sum() / 20 * 100
    return alpha


def alpha059(close, high, low):
    """
    Alpha#59: 同 alpha003 逻辑，窗口改为20
    """
    delay_close = close.shift(1)
    cond_eq = close == delay_close
    cond_up = close > delay_close
    term_up = close - np.minimum(low, delay_close)
    term_down = close - np.maximum(high, delay_close)
    inner = pd.DataFrame(
        np.where(cond_eq, 0.0, np.where(cond_up, term_up, term_down)),
        index=close.index, columns=close.columns
    )
    alpha = inner.rolling(20).sum()
    return alpha


def alpha060(close, high, low, volume):
    """
    Alpha#60: 同 alpha011 逻辑，窗口改为20
    """
    inner = ((close - low) - (high - close)) / (high - low) * volume
    alpha = inner.rolling(20).sum()
    return alpha


def alpha061(low, vwap, volume):
    """
    Alpha#61: (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),
                   RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80),8)),17))) * -1)
    """
    part1 = _decaylinear(vwap.diff(1), 12).rank(axis=1, pct=True)
    corr = low.rolling(8).corr(volume.rolling(80).mean())
    part2 = _decaylinear(corr.rank(axis=1, pct=True), 17).rank(axis=1, pct=True)
    alpha = -1 * np.maximum(part1, part2)
    return alpha


def alpha062(high, volume):
    """
    Alpha#62: (-1*CORR(HIGH,RANK(VOLUME),5))
    """
    rank_vol = volume.rank(axis=1, pct=True)
    alpha = -1 * high.rolling(5).corr(rank_vol)
    return alpha


def alpha063(close):
    """
    Alpha#63: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    alpha = up.ewm(alpha=1 / 6, adjust=False).mean() / diff.abs().ewm(alpha=1 / 6, adjust=False).mean() * 100
    return alpha


def alpha064(close, vwap, volume):
    """
    Alpha#64: (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),
                   RANK(DECAYLINEAR(TSMAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14))) * -1)
    """
    rank_vwap = vwap.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr1 = rank_vwap.rolling(4).corr(rank_vol)
    part1 = _decaylinear(corr1, 4).rank(axis=1, pct=True)

    rank_close = close.rank(axis=1, pct=True)
    rank_mean_vol60 = volume.rolling(60).mean().rank(axis=1, pct=True)
    corr2 = rank_close.rolling(4).corr(rank_mean_vol60)
    part2 = _decaylinear(corr2.rolling(13).max(), 14).rank(axis=1, pct=True)

    alpha = -1 * np.maximum(part1, part2)
    return alpha


def alpha065(close):
    """
    Alpha#65: MEAN(CLOSE,6)/CLOSE
    """
    alpha = close.rolling(6).mean() / close
    return alpha


def alpha066(close):
    """
    Alpha#66: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    """
    mean6 = close.rolling(6).mean()
    alpha = (close - mean6) / mean6 * 100
    return alpha


def alpha067(close):
    """
    Alpha#67: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    alpha = up.ewm(alpha=1 / 24, adjust=False).mean() / diff.abs().ewm(alpha=1 / 24, adjust=False).mean() * 100
    return alpha


def alpha068(high, low, volume):
    """
    Alpha#68: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    """
    mid = (high + low) / 2
    delay_mid = (high.shift(1) + low.shift(1)) / 2
    inner = (mid - delay_mid) * (high - low) / volume
    alpha = inner.ewm(alpha=2 / 15, adjust=False).mean()
    return alpha


def _dtm_dbm_open(open, high, low):
    """
    alpha069 专用 DTM/DBM：国君报告里基于 OPEN 的标准定义，
    和 alpha049/050/051 里基于 HIGH+LOW 展开的同名变量不是同一套，
    原始给定公式文本未展开这两个变量，此处按报告标准定义实现，建议核对。
    DTM = (OPEN<=DELAY(OPEN,1)) ? 0 : MAX(HIGH-OPEN, OPEN-DELAY(OPEN,1))
    DBM = (OPEN>=DELAY(OPEN,1)) ? 0 : MAX(OPEN-LOW, OPEN-DELAY(OPEN,1))
    """
    delay_open = open.shift(1)
    dtm = pd.DataFrame(
        np.where(open <= delay_open, 0.0, np.maximum(high - open, open - delay_open)),
        index=open.index, columns=open.columns
    )
    dbm = pd.DataFrame(
        np.where(open >= delay_open, 0.0, np.maximum(open - low, open - delay_open)),
        index=open.index, columns=open.columns
    )
    return dtm, dbm


def alpha069(open, high, low):
    """
    Alpha#69: SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)
              : (SUM(DTM,20)==SUM(DBM,20) ? 0 : (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20))
    """
    dtm, dbm = _dtm_dbm_open(open, high, low)
    sum_dtm = dtm.rolling(20).sum()
    sum_dbm = dbm.rolling(20).sum()

    alpha = pd.DataFrame(
        np.where(sum_dtm > sum_dbm, (sum_dtm - sum_dbm) / sum_dtm,
                 np.where(sum_dtm == sum_dbm, 0.0, (sum_dtm - sum_dbm) / sum_dbm)),
        index=open.index, columns=open.columns
    )
    return alpha


def alpha070(amount):
    """
    Alpha#70: STD(AMOUNT,6)
    """
    alpha = amount.rolling(6).std()
    return alpha


def alpha071(close):
    """
    Alpha#71: (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    """
    mean24 = close.rolling(24).mean()
    alpha = (close - mean24) / mean24 * 100
    return alpha


def alpha072(close, high, low):
    """
    Alpha#72: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    """
    tsmax_high6 = high.rolling(6).max()
    tsmin_low6 = low.rolling(6).min()
    inner = (tsmax_high6 - close) / (tsmax_high6 - tsmin_low6) * 100
    alpha = inner.ewm(alpha=1 / 15, adjust=False).mean()
    return alpha


def alpha073(close, vwap, volume):
    """
    Alpha#73: ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE,VOLUME,10),16),4),5)
                - RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3))) * -1)
    """
    corr1 = close.rolling(10).corr(volume)
    decay1 = _decaylinear(corr1, 16)
    decay2 = _decaylinear(decay1, 4)
    part1 = decay2.rolling(5).rank(pct=True)

    corr2 = vwap.rolling(4).corr(volume.rolling(30).mean())
    part2 = _decaylinear(corr2, 3).rank(axis=1, pct=True)

    alpha = -1 * (part1 - part2)
    return alpha


def alpha074(low, vwap, volume):
    """
    Alpha#74: (RANK(CORR(SUM((LOW*0.35)+(VWAP*0.65),20),SUM(MEAN(VOLUME,40),20),7))
               + RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))
    """
    weighted_price_sum = ((low * 0.35) + (vwap * 0.65)).rolling(20).sum()
    vol_mean_sum = volume.rolling(40).mean().rolling(20).sum()
    corr1 = weighted_price_sum.rolling(7).corr(vol_mean_sum)
    part1 = corr1.rank(axis=1, pct=True)

    rank_vwap = vwap.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr2 = rank_vwap.rolling(6).corr(rank_vol)
    part2 = corr2.rank(axis=1, pct=True)

    alpha = part1 + part2
    return alpha


def alpha075(open_df, close_df, benchmarkindex):
    """
    Alpha075：
    过去50个交易日内，
    大盘下跌且个股上涨的天数 / 大盘下跌的天数
    """
    # 股票数据对齐
    idx = close_df.index
    cols = close_df.columns.intersection(open_df.columns)

    open_aligned = open_df.reindex(index=idx,columns=cols)

    close_aligned = close_df.reindex(index=idx,columns=cols)

    # 指数数据对齐
    bench_open = benchmarkindex["open"].reindex(idx)
    bench_close = benchmarkindex["close"].reindex(idx)

    # 大盘当日下跌：Series，index=日期
    bench_down = (bench_close < bench_open).fillna(False)

    # 个股当日上涨：DataFrame，index=日期，columns=股票
    stock_up = (close_aligned > open_aligned).fillna(False)

    # 必须指定axis=0，让bench_down按日期与DataFrame的行对齐
    stock_up_when_bench_down = stock_up.astype(float).mul(bench_down.astype(float),axis=0)

    # 分子：过去50日，大盘跌且个股涨的次数
    numerator = stock_up_when_bench_down.rolling(window=50,min_periods=50).sum()

    # 分母：过去50日，大盘下跌次数
    denominator = bench_down.astype(float).rolling(window=50,min_periods=50).sum()
    denominator = denominator.replace(0, np.nan)

    # 同样必须指定axis=0，让分母按日期行广播
    alpha = numerator.div(denominator,axis=0)

    # 强制保证输出结构与close完全一致
    alpha = alpha.reindex(index=idx,columns=cols)
    return alpha

def alpha076(close, volume):
    """
    Alpha#76: STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)/MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)
    """
    ret_abs = (close / close.shift(1) - 1).abs()
    ratio = ret_abs / volume
    alpha = ratio.rolling(20).std() / ratio.rolling(20).mean()
    return alpha


def alpha077(high, low, vwap, volume):
    """
    Alpha#77: MIN(RANK(DECAYLINEAR((((HIGH+LOW)/2+HIGH)-(VWAP+HIGH)),20)),
                   RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),3),6)))
    (((HIGH+LOW)/2+HIGH)-(VWAP+HIGH)) 化简等于 (HIGH+LOW)/2 - VWAP
    """
    inner1 = (high + low) / 2 - vwap
    part1 = _decaylinear(inner1, 20).rank(axis=1, pct=True)

    mid = (high + low) / 2
    corr = mid.rolling(3).corr(volume.rolling(40).mean())
    part2 = _decaylinear(corr, 6).rank(axis=1, pct=True)

    alpha = np.minimum(part1, part2)
    return alpha


def alpha078(high, low, close):
    """
    Alpha#78: ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))
              / (0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    """
    typical = (high + low + close) / 3
    ma12_typical = typical.rolling(12).mean()
    abs_dev = (close - ma12_typical).abs()
    mean_abs_dev = abs_dev.rolling(12).mean()
    alpha = (typical - ma12_typical) / (0.015 * mean_abs_dev)
    return alpha


def alpha079(close):
    """
    Alpha#79: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    alpha = up.ewm(alpha=1 / 12, adjust=False).mean() / diff.abs().ewm(alpha=1 / 12, adjust=False).mean() * 100
    return alpha


def alpha080(volume):
    """
    Alpha#80: (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    """
    alpha = (volume - volume.shift(5)) / volume.shift(5) * 100
    return alpha


def alpha081(volume):
    """
    Alpha#81: SMA(VOLUME,21,2)
    """
    alpha = volume.ewm(alpha=2 / 21, adjust=False).mean()
    return alpha


def alpha082(close, high, low):
    """
    Alpha#82: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    """
    tsmax_high6 = high.rolling(6).max()
    tsmin_low6 = low.rolling(6).min()
    inner = (tsmax_high6 - close) / (tsmax_high6 - tsmin_low6) * 100
    alpha = inner.ewm(alpha=1 / 20, adjust=False).mean()
    return alpha


def alpha083(high, volume):
    """
    Alpha#83: (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    COVIANCE 视为 COVARIANCE 的笔误
    """
    rank_high = high.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    cov = rank_high.rolling(5).cov(rank_vol)
    alpha = -1 * cov.rank(axis=1, pct=True)
    return alpha


def alpha084(close, volume):
    """
    Alpha#84: 同 alpha043 逻辑，窗口改为20
    """
    delay_close = close.shift(1)
    signed_vol = pd.DataFrame(
        np.where(close > delay_close, volume, np.where(close < delay_close, -volume, 0.0)),
        index=close.index, columns=close.columns
    )
    alpha = signed_vol.rolling(20).sum()
    return alpha


def alpha085(close, volume):
    """
    Alpha#85: (TSRANK((VOLUME/MEAN(VOLUME,20)),20) * TSRANK((-1*DELTA(CLOSE,7)),8))
    """
    vol_ratio = volume / volume.rolling(20).mean()
    part1 = vol_ratio.rolling(20).rank(pct=True)
    part2 = (-1 * close.diff(7)).rolling(8).rank(pct=True)
    alpha = part1 * part2
    return alpha


def alpha086(close):
    """
    Alpha#86: 0.25 < (((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10)) ? -1
              : ((... < 0) ? 1 : (-1*(CLOSE-DELAY(CLOSE,1))))
    """
    term = ((close.shift(20) - close.shift(10)) / 10) - ((close.shift(10) - close) / 10)
    cond1 = term > 0.25
    cond2 = term < 0
    alpha = pd.DataFrame(
        np.where(cond1, -1.0, np.where(cond2, 1.0, -1 * (close - close.shift(1)))),
        index=close.index, columns=close.columns
    )
    return alpha


def alpha087(open, high, low, vwap):
    """
    Alpha#87: ((RANK(DECAYLINEAR(DELTA(VWAP,4),7)) +
                TSRANK(DECAYLINEAR((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2)),11),7)) * -1)
    (LOW*0.9)+(LOW*0.1) 字面化简后等于 LOW，按原始公式字面实现
    """
    part1 = _decaylinear(vwap.diff(4), 7).rank(axis=1, pct=True)
    weighted_low = (low * 0.9) + (low * 0.1)
    inner = (weighted_low - vwap) / (open - ((high + low) / 2))
    part2 = _decaylinear(inner, 11).rolling(7).rank(pct=True)
    alpha = -1 * (part1 + part2)
    return alpha


def alpha088(close):
    """
    Alpha#88: (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    """
    alpha = (close - close.shift(20)) / close.shift(20) * 100
    return alpha


def alpha089(close):
    """
    Alpha#89: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    """
    ema13 = close.ewm(alpha=2 / 13, adjust=False).mean()
    ema27 = close.ewm(alpha=2 / 27, adjust=False).mean()
    diff = ema13 - ema27
    signal = diff.ewm(alpha=2 / 10, adjust=False).mean()
    alpha = 2 * (diff - signal)
    return alpha


def alpha090(volume, vwap):
    """
    Alpha#90: (RANK(CORR(RANK(VWAP),RANK(VOLUME),5)) * -1)
    """
    rank_vwap = vwap.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr = rank_vwap.rolling(5).corr(rank_vol)
    alpha = -1 * corr.rank(axis=1, pct=True)
    return alpha


def alpha091(close, low, volume):
    """
    Alpha#91: ((RANK((CLOSE-TSMAX(CLOSE,5)))*RANK(CORR(MEAN(VOLUME,40),LOW,5))) * -1)
    """
    part1 = (close - close.rolling(5).max()).rank(axis=1, pct=True)
    corr = volume.rolling(40).mean().rolling(5).corr(low)
    part2 = corr.rank(axis=1, pct=True)
    alpha = -1 * part1 * part2
    return alpha


def alpha092(close, vwap, volume):
    """
    Alpha#92: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),
                   TSRANK(DECAYLINEAR(ABS(CORR(MEAN(VOLUME,180),CLOSE,13)),5),15)) * -1)
    """
    weighted_price = (close * 0.35) + (vwap * 0.65)
    part1 = _decaylinear(weighted_price.diff(2), 3).rank(axis=1, pct=True)
    corr = volume.rolling(180).mean().rolling(13).corr(close).abs()
    part2 = _decaylinear(corr, 5).rolling(15).rank(pct=True)
    alpha = -1 * np.maximum(part1, part2)
    return alpha


def alpha093(open, high, low):
    """
    Alpha#93: SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    复用 alpha069 的 DBM（基于OPEN的标准定义）
    """
    _, dbm = _dtm_dbm_open(open, high, low)
    alpha = dbm.rolling(20).sum()
    return alpha


def alpha094(close, volume):
    """
    Alpha#94: 同 alpha043 逻辑，窗口改为30
    """
    delay_close = close.shift(1)
    signed_vol = pd.DataFrame(
        np.where(close > delay_close, volume, np.where(close < delay_close, -volume, 0.0)),
        index=close.index, columns=close.columns
    )
    alpha = signed_vol.rolling(30).sum()
    return alpha


def alpha095(amount):
    """
    Alpha#95: STD(AMOUNT,20)
    """
    alpha = amount.rolling(20).std()
    return alpha


def alpha096(close, high, low):
    """
    Alpha#96: SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    """
    tsmin_low9 = low.rolling(9).min()
    tsmax_high9 = high.rolling(9).max()
    rsv = (close - tsmin_low9) / (tsmax_high9 - tsmin_low9) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    alpha = d
    return alpha


def alpha097(volume):
    """
    Alpha#97: STD(VOLUME,10)
    """
    alpha = volume.rolling(10).std()
    return alpha


def alpha098(close):
    """
    Alpha#98: ((DELTA(SUM(CLOSE,100)/100,100)/DELAY(CLOSE,100)) <= 0.05
               ? (-1*(CLOSE-TSMIN(CLOSE,100))) : (-1*DELTA(CLOSE,3)))
    """
    mean100 = close.rolling(100).sum() / 100
    ratio = mean100.diff(100) / close.shift(100)
    cond = ratio <= 0.05
    alpha = pd.DataFrame(
        np.where(cond, -1 * (close - close.rolling(100).min()), -1 * close.diff(3)),
        index=close.index, columns=close.columns
    )
    return alpha


def alpha099(close, volume):
    """
    Alpha#99: (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    """
    rank_close = close.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    cov = rank_close.rolling(5).cov(rank_vol)
    alpha = -1 * cov.rank(axis=1, pct=True)
    return alpha


def alpha100(volume):
    """
    Alpha#100: STD(VOLUME,20)
    """
    alpha = volume.rolling(20).std()
    return alpha


def alpha101(close, high, vwap, volume):
    """
    Alpha#101: ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15)) <
                 RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11))) * -1)
    """
    vol_mean_sum = volume.rolling(30).mean().rolling(37).sum()
    corr1 = close.rolling(15).corr(vol_mean_sum)
    part1 = corr1.rank(axis=1, pct=True)

    weighted_price = (high * 0.1) + (vwap * 0.9)
    rank_price = weighted_price.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr2 = rank_price.rolling(11).corr(rank_vol)
    part2 = corr2.rank(axis=1, pct=True)

    alpha = -1 * (part1 < part2).astype(float)
    return alpha


def alpha102(volume):
    """
    Alpha#102: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    diff = volume - volume.shift(1)
    up = diff.clip(lower=0)
    alpha = up.ewm(alpha=1 / 6, adjust=False).mean() / diff.abs().ewm(alpha=1 / 6, adjust=False).mean() * 100
    return alpha


def alpha103(low):
    """
    Alpha#103: ((20-LOWDAY(LOW,20))/20)*100
    """
    lowday = _lowday(low, 20)
    alpha = (20 - lowday) / 20 * 100
    return alpha


def alpha104(close, high, volume):
    """
    Alpha#104: (-1 * (DELTA(CORR(HIGH,VOLUME,5),5) * RANK(STD(CLOSE,20))))
    """
    corr = high.rolling(5).corr(volume)
    part1 = corr.diff(5)
    part2 = close.rolling(20).std().rank(axis=1, pct=True)
    alpha = -1 * part1 * part2
    return alpha


def alpha105(open, volume):
    """
    Alpha#105: (-1 * CORR(RANK(OPEN),RANK(VOLUME),10))
    """
    rank_open = open.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    alpha = -1 * rank_open.rolling(10).corr(rank_vol)
    return alpha


def alpha106(close):
    """
    Alpha#106: CLOSE-DELAY(CLOSE,20)
    """
    alpha = close - close.shift(20)
    return alpha


def alpha107(open, high, low, close):
    """
    Alpha#107: (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
    """
    part1 = -1 * (open - high.shift(1)).rank(axis=1, pct=True)
    part2 = (open - close.shift(1)).rank(axis=1, pct=True)
    part3 = (open - low.shift(1)).rank(axis=1, pct=True)
    alpha = part1 * part2 * part3
    return alpha


def alpha108(high, vwap, volume):
    """
    Alpha#108: ((RANK((HIGH-TSMIN(HIGH,2)))^RANK(CORR(VWAP,MEAN(VOLUME,120),6))) * -1)
    先RANK底数收敛到[0,1]避免负数底数配分数指数
    """
    base = (high - high.rolling(2).min()).rank(axis=1, pct=True)
    exponent = vwap.rolling(6).corr(volume.rolling(120).mean()).rank(axis=1, pct=True)
    alpha = -1 * (base ** exponent)
    return alpha


def alpha109(high, low):
    """
    Alpha#109: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    """
    diff = high - low
    sma1 = diff.ewm(alpha=2 / 10, adjust=False).mean()
    sma2 = sma1.ewm(alpha=2 / 10, adjust=False).mean()
    alpha = sma1 / sma2
    return alpha


def alpha110(close, high, low):
    """
    Alpha#110: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    """
    delay_close = close.shift(1)
    up = (high - delay_close).clip(lower=0)
    down = (delay_close - low).clip(lower=0)
    alpha = up.rolling(20).sum() / down.rolling(20).sum() * 100
    return alpha


def alpha111(close, high, low, volume):
    """
    Alpha#111: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    VOL 即 VOLUME
    """
    inner = volume * ((close - low) - (high - close)) / (high - low)
    alpha = inner.ewm(alpha=2 / 11, adjust=False).mean() - inner.ewm(alpha=2 / 4, adjust=False).mean()
    return alpha


def alpha112(close):
    """
    Alpha#112: (SUM(up,12)-SUM(down,12))/(SUM(up,12)+SUM(down,12))*100
    up=max(diff,0), down=max(-diff,0)
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    sum_up = up.rolling(12).sum()
    sum_down = down.rolling(12).sum()
    alpha = (sum_up - sum_down) / (sum_up + sum_down) * 100
    return alpha


def alpha113(close, volume):
    """
    Alpha#113: (-1 * ((RANK((SUM(DELAY(CLOSE,5),20)/20))*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))
    """
    part1 = (close.shift(5).rolling(20).sum() / 20).rank(axis=1, pct=True)
    part2 = close.rolling(2).corr(volume)
    sum5 = close.rolling(5).sum()
    sum20 = close.rolling(20).sum()
    part3 = sum5.rolling(2).corr(sum20).rank(axis=1, pct=True)
    alpha = -1 * (part1 * part2 * part3)
    return alpha


def alpha114(close, high, low, vwap, volume):
    """
    Alpha#114: (RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2)) * RANK(RANK(VOLUME)))
               / (((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE))
    """
    ratio = (high - low) / (close.rolling(5).sum() / 5)
    part1 = ratio.shift(2).rank(axis=1, pct=True)
    part2 = volume.rank(axis=1, pct=True).rank(axis=1, pct=True)
    denom = ratio / (vwap - close)
    alpha = (part1 * part2) / denom
    return alpha


def alpha115(close, high, low, volume):
    """
    Alpha#115: (RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))
                ^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))
    """
    weighted_price = (high * 0.9) + (close * 0.1)
    base = weighted_price.rolling(10).corr(volume.rolling(30).mean()).rank(axis=1, pct=True)
    tsrank_mid = ((high + low) / 2).rolling(4).rank(pct=True)
    tsrank_vol = volume.rolling(10).rank(pct=True)
    exponent = tsrank_mid.rolling(7).corr(tsrank_vol).rank(axis=1, pct=True)
    alpha = base ** exponent
    return alpha


def alpha116(close):
    """
    Alpha#116: REGBETA(CLOSE,SEQUENCE,20)
    过去20日 CLOSE 对时间序号[1..20]做回归，取斜率
    """
    alpha = _regbeta_seq(close, 20)
    return alpha


def alpha117(close, high, low, volume):
    """
    Alpha#117: ((TSRANK(VOLUME,32)*(1-TSRANK(((CLOSE+HIGH)-LOW),16)))*(1-TSRANK(RET,32)))
    """
    ret = close.pct_change()
    part1 = volume.rolling(32).rank(pct=True)
    part2 = 1 - ((close + high) - low).rolling(16).rank(pct=True)
    part3 = 1 - ret.rolling(32).rank(pct=True)
    alpha = part1 * part2 * part3
    return alpha


def alpha118(open, high, low):
    """
    Alpha#118: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    """
    alpha = (high - open).rolling(20).sum() / (open - low).rolling(20).sum() * 100
    return alpha


def alpha119(open, vwap, volume):
    """
    Alpha#119: (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))
                -RANK(DECAYLINEAR(TSRANK(TSMIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
    """
    vol_mean_sum = volume.rolling(5).mean().rolling(26).sum()
    corr1 = vwap.rolling(5).corr(vol_mean_sum)
    part1 = _decaylinear(corr1, 7).rank(axis=1, pct=True)

    rank_open = open.rank(axis=1, pct=True)
    rank_vol_mean15 = volume.rolling(15).mean().rank(axis=1, pct=True)
    corr2 = rank_open.rolling(21).corr(rank_vol_mean15)
    tsmin_corr2 = corr2.rolling(9).min()
    tsrank_val = tsmin_corr2.rolling(7).rank(pct=True)
    part2 = _decaylinear(tsrank_val, 8).rank(axis=1, pct=True)

    alpha = part1 - part2
    return alpha


def alpha120(close, vwap):
    """
    Alpha#120: (RANK((VWAP-CLOSE)) / RANK((VWAP+CLOSE)))
    """
    part1 = (vwap - close).rank(axis=1, pct=True)
    part2 = (vwap + close).rank(axis=1, pct=True)
    alpha = part1 / part2
    return alpha


def alpha121(vwap, volume):
    """
    Alpha#121: ((RANK((VWAP-TSMIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3)) * -1)
    """
    base = (vwap - vwap.rolling(12).min()).rank(axis=1, pct=True)
    tsrank_vwap20 = vwap.rolling(20).rank(pct=True)
    tsrank_volmean2 = volume.rolling(60).mean().rolling(2).rank(pct=True)
    corr = tsrank_vwap20.rolling(18).corr(tsrank_volmean2)
    exponent = corr.rolling(3).rank(pct=True)
    alpha = -1 * (base ** exponent)
    return alpha


def alpha122(close):
    """
    Alpha#122: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(...,1))/DELAY(...,1)
    """
    log_close = np.log(close)
    sma1 = log_close.ewm(alpha=2 / 13, adjust=False).mean()
    sma2 = sma1.ewm(alpha=2 / 13, adjust=False).mean()
    sma3 = sma2.ewm(alpha=2 / 13, adjust=False).mean()
    alpha = (sma3 - sma3.shift(1)) / sma3.shift(1)
    return alpha


def alpha123(high, low, volume):
    """
    Alpha#123: ((RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9))<RANK(CORR(LOW,VOLUME,6))) * -1)
    """
    sum_mid = ((high + low) / 2).rolling(20).sum()
    sum_vol_mean = volume.rolling(60).mean().rolling(20).sum()
    part1 = sum_mid.rolling(9).corr(sum_vol_mean).rank(axis=1, pct=True)
    part2 = low.rolling(6).corr(volume).rank(axis=1, pct=True)
    alpha = -1 * (part1 < part2).astype(float)
    return alpha


def alpha124(close, vwap):
    """
    Alpha#124: (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
    """
    rank_tsmax = close.rolling(30).max().rank(axis=1, pct=True)
    denom = _decaylinear(rank_tsmax, 2)
    alpha = (close - vwap) / denom
    return alpha


def alpha125(close, vwap, volume):
    """
    Alpha#125: (RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,80),17),20))
                / RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))
    """
    corr = vwap.rolling(17).corr(volume.rolling(80).mean())
    part1 = _decaylinear(corr, 20).rank(axis=1, pct=True)
    weighted_price = (close * 0.5) + (vwap * 0.5)
    part2 = _decaylinear(weighted_price.diff(3), 16).rank(axis=1, pct=True)
    alpha = part1 / part2
    return alpha


def alpha126(close, high, low):
    """
    Alpha#126: (CLOSE+HIGH+LOW)/3
    """
    alpha = (close + high + low) / 3
    return alpha


def alpha127(close):
    """
    Alpha#127: (MEAN((100*(CLOSE-TSMAX(CLOSE,12))/TSMAX(CLOSE,12))^2,N))^(1/2)
    原文外层 MEAN 未给窗口，按内层 TSMAX 的窗口一致取12
    """
    tsmax12 = close.rolling(12).max()
    inner = (100 * (close - tsmax12) / tsmax12) ** 2
    alpha = inner.rolling(12).mean() ** 0.5
    return alpha


def alpha128(high, low, close, volume):
    """
    Alpha#128: 100-(100/(1+SUM(up_mf,14)/SUM(down_mf,14)))，即经典 Money Flow Index (MFI)
    typical=(HIGH+LOW+CLOSE)/3，up_mf/down_mf 按 typical 涨跌分流 typical*VOLUME
    """
    typical = (high + low + close) / 3
    delay_typical = typical.shift(1)
    money_flow = typical * volume
    up_mf = pd.DataFrame(np.where(typical > delay_typical, money_flow, 0.0), index=close.index, columns=close.columns)
    down_mf = pd.DataFrame(np.where(typical < delay_typical, money_flow, 0.0), index=close.index, columns=close.columns)
    ratio = up_mf.rolling(14).sum() / down_mf.rolling(14).sum()
    alpha = 100 - (100 / (1 + ratio))
    return alpha


def alpha129(close):
    """
    Alpha#129: SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    """
    diff = close - close.shift(1)
    down = pd.DataFrame(np.where(diff < 0, diff.abs(), 0.0), index=close.index, columns=close.columns)
    alpha = down.rolling(12).sum()
    return alpha


def alpha130(high, low, vwap, volume):
    """
    Alpha#130: (RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10))
                /RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
    """
    mid = (high + low) / 2
    corr1 = mid.rolling(9).corr(volume.rolling(40).mean())
    part1 = _decaylinear(corr1, 10).rank(axis=1, pct=True)

    rank_vwap = vwap.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    corr2 = rank_vwap.rolling(7).corr(rank_vol)
    part2 = _decaylinear(corr2, 3).rank(axis=1, pct=True)

    alpha = part1 / part2
    return alpha


def alpha131(close, vwap, volume):
    """
    Alpha#131: (RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
    原文 DELAT 视为 DELTA 的笔误
    """
    base = vwap.diff(1).rank(axis=1, pct=True)
    corr = close.rolling(18).corr(volume.rolling(50).mean())
    exponent = corr.rolling(18).rank(pct=True)
    alpha = base ** exponent
    return alpha


def alpha132(amount):
    """
    Alpha#132: MEAN(AMOUNT,20)
    """
    alpha = amount.rolling(20).mean()
    return alpha


def alpha133(high, low):
    """
    Alpha#133: ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    """
    highday = _highday(high, 20)
    lowday = _lowday(low, 20)
    alpha = (20 - highday) / 20 * 100 - (20 - lowday) / 20 * 100
    return alpha


def alpha134(close, volume):
    """
    Alpha#134: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    """
    alpha = (close - close.shift(12)) / close.shift(12) * volume
    return alpha


def alpha135(close):
    """
    Alpha#135: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    """
    ratio = close / close.shift(20)
    delayed = ratio.shift(1)
    alpha = delayed.ewm(alpha=1 / 20, adjust=False).mean()
    return alpha


def alpha136(open, close, volume):
    """
    Alpha#136: ((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))
    """
    ret = close.pct_change()
    part1 = -1 * ret.diff(3).rank(axis=1, pct=True)
    part2 = open.rolling(10).corr(volume)
    alpha = part1 * part2
    return alpha


def alpha137(close, open, high, low):
    """
    Alpha#137: 与 alpha055 是同一套三分支真实波幅结构，区别是这里不做20日滚动求和，直接输出逐日值
    """
    delay_close = close.shift(1)
    delay_open = open.shift(1)
    delay_low = low.shift(1)

    term_a = (high - delay_close).abs()
    term_b = (low - delay_close).abs()
    term_c = (high - delay_low).abs()
    term_d = (delay_close - delay_open).abs()

    cond1 = (term_a > term_b) & (term_a > term_c)
    cond2 = (term_b > term_c) & (term_b > term_a)

    denom1 = term_a + term_b / 2 + term_d / 4
    denom2 = term_b + term_a / 2 + term_d / 4
    denom3 = term_c + term_d / 4

    denom = pd.DataFrame(
        np.where(cond1, denom1, np.where(cond2, denom2, denom3)),
        index=close.index, columns=close.columns
    )

    numerator = 16 * (close - delay_close + (close - open) / 2 + delay_close - delay_open)
    max_term = np.maximum(term_a, term_b)

    alpha = numerator / denom * max_term
    return alpha


def alpha138(low, vwap, volume):
    """
    Alpha#138: ((RANK(DECAYLINEAR(DELTA(((LOW*0.7)+(VWAP*0.3)),3),20))
                -TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7)) * -1)
    """
    weighted_price = (low * 0.7) + (vwap * 0.3)
    part1 = _decaylinear(weighted_price.diff(3), 20).rank(axis=1, pct=True)

    tsrank_low8 = low.rolling(8).rank(pct=True)
    tsrank_volmean17 = volume.rolling(60).mean().rolling(17).rank(pct=True)
    corr = tsrank_low8.rolling(5).corr(tsrank_volmean17)
    tsrank_corr19 = corr.rolling(19).rank(pct=True)
    decay16 = _decaylinear(tsrank_corr19, 16)
    part2 = decay16.rolling(7).rank(pct=True)

    alpha = -1 * (part1 - part2)
    return alpha


def alpha139(open, volume):
    """
    Alpha#139: (-1 * CORR(OPEN,VOLUME,10))
    """
    alpha = -1 * open.rolling(10).corr(volume)
    return alpha


def alpha140(open, high, low, close, volume):
    """
    Alpha#140: MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),
                    TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
    """
    inner = (open.rank(axis=1, pct=True) + low.rank(axis=1, pct=True)) \
        - (high.rank(axis=1, pct=True) + close.rank(axis=1, pct=True))
    part1 = _decaylinear(inner, 8).rank(axis=1, pct=True)

    tsrank_close8 = close.rolling(8).rank(pct=True)
    tsrank_volmean20 = volume.rolling(60).mean().rolling(20).rank(pct=True)
    corr = tsrank_close8.rolling(8).corr(tsrank_volmean20)
    decay7 = _decaylinear(corr, 7)
    part2 = decay7.rolling(3).rank(pct=True)

    alpha = np.minimum(part1, part2)
    return alpha


def alpha141(high, volume):
    """
    Alpha#141: (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9)) * -1)
    """
    rank_high = high.rank(axis=1, pct=True)
    rank_vol_mean15 = volume.rolling(15).mean().rank(axis=1, pct=True)
    corr = rank_high.rolling(9).corr(rank_vol_mean15)
    alpha = -1 * corr.rank(axis=1, pct=True)
    return alpha


def alpha142(close, volume):
    """
    Alpha#142: (((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))
    """
    part1 = -1 * close.rolling(10).rank(pct=True).rank(axis=1, pct=True)
    part2 = close.diff(1).diff(1).rank(axis=1, pct=True)
    vol_ratio = volume / volume.rolling(20).mean()
    part3 = vol_ratio.rolling(5).rank(pct=True).rank(axis=1, pct=True)
    alpha = part1 * part2 * part3
    return alpha


def alpha143(close):
    """
    Alpha#143: CLOSE>DELAY(CLOSE,1) ? (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF : SELF
    SELF 是因子自身的前一日值，本质是"只在上涨日复利、下跌日不变"的累计乘积过程，
    初始值（第一个有效交易日）按惯例取1
    """
    ret = close.pct_change()
    up = close > close.shift(1)
    multiplier = pd.DataFrame(np.where(up, 1 + ret, 1.0), index=close.index, columns=close.columns)
    alpha = multiplier.cumprod()
    return alpha


def alpha144(close, amount):
    """
    Alpha#144: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    SUMIF(X,N,cond)：过去N日内仅对满足cond的日子累加X
    """
    delay_close = close.shift(1)
    cond = close < delay_close
    ratio = (close / delay_close - 1).abs() / amount
    sumif_val = pd.DataFrame(np.where(cond, ratio, 0.0), index=close.index, columns=close.columns).rolling(20).sum()
    count_val = cond.astype(float).rolling(20).sum()
    alpha = sumif_val / count_val
    return alpha


def alpha145(volume):
    """
    Alpha#145: (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    """
    alpha = (volume.rolling(9).mean() - volume.rolling(26).mean()) / volume.rolling(12).mean() * 100
    return alpha


def alpha146(close):
    """
    Alpha#146: MEAN(dev,20)*dev / SMA((ret-dev)^2,60)，其中 ret=(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)，
    dev = ret - SMA(ret,61,2)
    化简：ret-dev = SMA(ret,61,2)，所以分母 = SMA(SMA(ret,61,2)^2, 60)；
    原文分母 "SMA(...,60)" 只给了一个窗口参数没有平滑系数M，按 MEAN(...,60) 处理
    """
    ret = close.pct_change()
    sma_ret = ret.ewm(alpha=2 / 61, adjust=False).mean()
    dev = ret - sma_ret
    numerator = dev.rolling(20).mean() * dev
    denom = (sma_ret ** 2).rolling(60).mean()
    alpha = numerator / denom
    return alpha


def alpha147(close):
    """
    Alpha#147: REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    """
    mean12 = close.rolling(12).mean()
    alpha = _regbeta_seq(mean12, 12)
    return alpha


def alpha148(open, volume):
    """
    Alpha#148: ((RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14)))) * -1)
    """
    vol_mean_sum = volume.rolling(60).mean().rolling(9).sum()
    part1 = open.rolling(6).corr(vol_mean_sum).rank(axis=1, pct=True)
    part2 = (open - open.rolling(14).min()).rank(axis=1, pct=True)
    alpha = -1 * (part1 < part2).astype(float)
    return alpha


def alpha149(benchmarkindex, close):
    """
    Alpha#149: REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,benchmarkindexCLOSE<DELAY(benchmarkindexCLOSE,1)),FILTER(benchmarkindexCLOSE/DELAY(benchmarkindexCLOSE,1)-1,benchmarkindexCLOSE<DELAY(benchmarkindexCLOSE,1)),252)
    """
    bench_close = benchmarkindex["close"].reindex(close.index)
    stock_ret = close / close.shift(1) - 1
    bench_ret = bench_close / bench_close.shift(1) - 1
    bench_down = (bench_close < bench_close.shift(1)).fillna(False)
    alpha = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    # FILTER 会先删除非基准下跌日，因此 252 指最近 252 个过滤后的
    # 有效观测，而不是最近 252 个日历交易日。
    for stock in close.columns:
        valid = (
            bench_down
            & bench_ret.notna()
            & stock_ret[stock].notna()
        )
        filtered_stock_ret = stock_ret.loc[valid, stock]
        filtered_bench_ret = bench_ret.loc[valid]

        covariance = filtered_stock_ret.rolling(
            252, min_periods=252
        ).cov(filtered_bench_ret)
        variance = filtered_bench_ret.rolling(
            252, min_periods=252
        ).var()
        beta = covariance / variance.replace(0, np.nan)
        alpha.loc[beta.index, stock] = beta

    return alpha


def alpha150(close, high, low, volume):
    """
    Alpha#150: (CLOSE+HIGH+LOW)/3*VOLUME
    """
    alpha = (close + high + low) / 3 * volume
    return alpha


def alpha151(close):
    """
    Alpha#151: SMA(CLOSE-DELAY(CLOSE,20),20,1)
    """
    alpha = (close - close.shift(20)).ewm(alpha=1 / 20, adjust=False).mean()
    return alpha


def alpha152(close):
    """
    Alpha#152: SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)
                   -MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    """
    ratio9 = close / close.shift(9)
    delayed_ratio9 = ratio9.shift(1)
    inner_sma = delayed_ratio9.ewm(alpha=1 / 9, adjust=False).mean()
    delayed_inner_sma = inner_sma.shift(1)
    mean12 = delayed_inner_sma.rolling(12).mean()
    mean26 = delayed_inner_sma.rolling(26).mean()
    diff = mean12 - mean26
    alpha = diff.ewm(alpha=1 / 9, adjust=False).mean()
    return alpha


def alpha153(close):
    """
    Alpha#153: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    """
    alpha = (close.rolling(3).mean() + close.rolling(6).mean()
             + close.rolling(12).mean() + close.rolling(24).mean()) / 4
    return alpha


def alpha154(vwap, volume):
    """
    Alpha#154: (((VWAP-TSMIN(VWAP,16))) < (CORR(VWAP,MEAN(VOLUME,180),18)))
    比较结果转成 1.0/0.0
    """
    part1 = vwap - vwap.rolling(16).min()
    part2 = vwap.rolling(18).corr(volume.rolling(180).mean())
    alpha = (part1 < part2).astype(float)
    return alpha


def alpha155(volume):
    """
    Alpha#155: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    """
    ema13 = volume.ewm(alpha=2 / 13, adjust=False).mean()
    ema27 = volume.ewm(alpha=2 / 27, adjust=False).mean()
    diff = ema13 - ema27
    signal = diff.ewm(alpha=2 / 10, adjust=False).mean()
    alpha = diff - signal
    return alpha


def alpha156(open, low, vwap):
    """
    Alpha#156: (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),
                    RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3))) * -1)
    """
    part1 = _decaylinear(vwap.diff(5), 3).rank(axis=1, pct=True)

    weighted_price = (open * 0.15) + (low * 0.85)
    inner = (weighted_price.diff(2) / weighted_price) * -1
    part2 = _decaylinear(inner, 3).rank(axis=1, pct=True)

    alpha = -1 * np.maximum(part1, part2)
    return alpha


def alpha157(close):
    """
    Alpha#157: (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)
                +TSRANK(DELAY((-1*RET),6),5))
    SUM(...,1) 和 PROD(...,1) 都是窗口为1的恒等操作，化简后省略；
    DELTA((CLOSE-1),5) 等于 DELTA(CLOSE,5)（常数-1在做差时抵消）
    """
    ret = close.pct_change()
    delta5 = close.diff(5)
    r1 = delta5.rank(axis=1, pct=True)
    r2 = (-1 * r1).rank(axis=1, pct=True)
    r3 = r2.rank(axis=1, pct=True)
    tsmin2 = r3.rolling(2).min()
    log_val = np.log(tsmin2)
    r4 = log_val.rank(axis=1, pct=True)
    r5 = r4.rank(axis=1, pct=True)
    part1 = r5.rolling(5).min()

    part2 = (-1 * ret).shift(6).rolling(5).rank(pct=True)

    alpha = part1 + part2
    return alpha


def alpha158(close, high, low):
    """
    Alpha#158: ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    两个 SMA(CLOSE,15,2) 相互抵消，字面化简后等于 (HIGH-LOW)/CLOSE
    """
    alpha = (high - low) / close
    return alpha


def alpha159(close, high, low):
    """
    Alpha#159: 按原文字面括号实现——分子是"当日CLOSE减去N日累计的MIN(LOW,DELAY(CLOSE,1))"，
    不是先逐日算差再求和（用户已两次确认原文括号就是这样写的，不是转录误差）。
    分母 SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),N) 是标准的N日真实波幅累计。
    三个周期(6,12,24)按 12*24/6*24/6*24 加权后除以权重总和(6*12+6*24+12*24)。
    原文 "HGIH" 是 HIGH 的笔误。
    """
    delay_close = close.shift(1)
    min_lc = np.minimum(low, delay_close)
    max_hc = np.maximum(high, delay_close)
    tr = max_hc - min_lc

    def term(window, weight):
        numerator = close - min_lc.rolling(window).sum()
        denom = tr.rolling(window).sum()
        return numerator / denom * weight

    t6 = term(6, 12 * 24)
    t12 = term(12, 6 * 24)
    t24 = term(24, 6 * 24)

    alpha = (t6 + t12 + t24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
    return alpha


def alpha160(close):
    """
    Alpha#160: SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    """
    std20 = close.rolling(20).std()
    delay_close = close.shift(1)
    inner = pd.DataFrame(np.where(close <= delay_close, std20, 0.0), index=close.index, columns=close.columns)
    alpha = inner.ewm(alpha=1 / 20, adjust=False).mean()
    return alpha


def alpha161(close, high, low):
    """
    Alpha#161: MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    经典 True Range 的12日均值（类ATR-12）
    """
    delay_close = close.shift(1)
    tr = np.maximum(np.maximum(high - low, (delay_close - high).abs()), (delay_close - low).abs())
    alpha = tr.rolling(12).mean()
    return alpha


def alpha162(close):
    """
    Alpha#162: 对12日RSI式指标(SMA(MAX(diff,0),12,1)/SMA(ABS(diff),12,1)*100)再做12日Stochastic式归一化：
    (RSI-TSMIN(RSI,12))/(TSMAX(RSI,12)-TSMIN(RSI,12))
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    rsi = up.ewm(alpha=1 / 12, adjust=False).mean() / diff.abs().ewm(alpha=1 / 12, adjust=False).mean() * 100
    rsi_min = rsi.rolling(12).min()
    rsi_max = rsi.rolling(12).max()
    alpha = (rsi - rsi_min) / (rsi_max - rsi_min)
    return alpha


def alpha163(close, high, vwap, volume):
    """
    Alpha#163: RANK((((-1*RET)*MEAN(VOLUME,20))*VWAP)*(HIGH-CLOSE))
    """
    ret = close.pct_change()
    inner = (-1 * ret) * volume.rolling(20).mean() * vwap * (high - close)
    alpha = inner.rank(axis=1, pct=True)
    return alpha


def alpha164(close, high, low):
    """
    Alpha#164: SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-TSMIN(x,12))/(HIGH-LOW)*100,13,2)
    """
    delay_close = close.shift(1)
    diff = close - delay_close
    x = pd.DataFrame(np.where(close > delay_close, 1 / diff, 1.0), index=close.index, columns=close.columns)
    inner = (x - x.rolling(12).min()) / (high - low) * 100
    alpha = inner.ewm(alpha=2 / 13, adjust=False).mean()
    return alpha


def alpha165(close):
    """
    Alpha#165: MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    原文外层 MAX/MIN(SUMAC(...)) 都没给窗口，SUMAC=累计和(cumsum)。这本质是经典 Hurst 指数计算里的
    R/S(重标极差)统计量：窗口内对均值的离差做cumsum，取cumsum的极差(max-min)除以窗口内标准差。
    按公式里唯一出现的窗口48，把 R/S 统计量实现成48日滚动窗口版本。
    """
    window = 48

    def rs_stat(x):
        if np.isnan(x).any():
            return np.nan
        dev = x - x.mean()
        cum = np.cumsum(dev)
        std = x.std()
        if std == 0:
            return np.nan
        return (cum.max() - cum.min()) / std

    alpha = close.rolling(window).apply(rs_stat, raw=True)
    return alpha


def alpha166(close):
    """
    Alpha#166: 原文这条公式OCR损坏严重（"-20*(20-1)^1.5*SUM(...)/((20-1)*(20-2)(SUM(...)^2,20))^1.5"，
    括号明显残缺）。若把 SUM(...) 字面理解为对"收益率-均值"的一次方求和，结果按算术平均定义恒等于0
    （离差之和必为0），因子会变成永远输出0，不可能是原意，判断原文丢失了立方项。
    按标准"20日收益率偏度(skewness)"统计量的构造实现：
        numerator = -20*(19)^1.5 * SUM((ret-MEAN(ret,20))^3, 20)
        denominator = 19*18*(SUM(ret^2,20))^1.5
    请务必对照原始研报核实这条，这是本批次里我最不确定的一个转写。
    """
    ret = close.pct_change()
    mean_ret = ret.rolling(20).mean()
    dev = ret - mean_ret
    numerator = -20 * (19 ** 1.5) * (dev ** 3).rolling(20).sum()
    denom = 19 * 18 * (ret ** 2).rolling(20).sum() ** 1.5
    alpha = numerator / denom
    return alpha


def alpha167(close):
    """
    Alpha#167: SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    """
    diff = close - close.shift(1)
    up = diff.clip(lower=0)
    alpha = up.rolling(12).sum()
    return alpha


def alpha168(volume):
    """
    Alpha#168: (-1*VOLUME/MEAN(VOLUME,20))
    """
    alpha = -1 * volume / volume.rolling(20).mean()
    return alpha


def alpha169(close):
    """
    Alpha#169: SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    结构上和 alpha152 相同，只是内层用 CLOSE-DELAY(CLOSE,1)（差值）而非比值
    """
    diff = close - close.shift(1)
    inner_sma = diff.ewm(alpha=1 / 9, adjust=False).mean()
    delayed_inner_sma = inner_sma.shift(1)
    mean12 = delayed_inner_sma.rolling(12).mean()
    mean26 = delayed_inner_sma.rolling(26).mean()
    diff2 = mean12 - mean26
    alpha = diff2.ewm(alpha=1 / 10, adjust=False).mean()
    return alpha


def alpha170(close, high, vwap, volume):
    """
    Alpha#170: ((((RANK((1/CLOSE))*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))
                -RANK((VWAP-DELAY(VWAP,5))))
    """
    part1 = (1 / close).rank(axis=1, pct=True) * volume / volume.rolling(20).mean()
    part2 = (high * (high - close).rank(axis=1, pct=True)) / (high.rolling(5).sum() / 5)
    part3 = (vwap - vwap.shift(5)).rank(axis=1, pct=True)
    alpha = (part1 * part2) - part3
    return alpha


def alpha171(open, close, high, low):
    """
    Alpha#171: ((-1*((LOW-CLOSE)*(OPEN^5)))/((CLOSE-HIGH)*(CLOSE^5)))
    """
    numerator = -1 * (low - close) * (open ** 5)
    denominator = (close - high) * (close ** 5)
    alpha = numerator / denominator
    return alpha


def alpha172(close, high, low):
    """
    Alpha#172: 经典 ADX（Average Directional Index）结构：
    HD=HIGH-DELAY(HIGH,1)，LD=DELAY(LOW,1)-LOW，TR=真实波幅；
    +DI=100*SUM(HD筛选后,14)/SUM(TR,14)，-DI=100*SUM(LD筛选后,14)/SUM(TR,14)；
    DX=ABS(+DI - -DI)/(+DI + -DI)*100，ADX=MEAN(DX,6)
    """
    delay_high = high.shift(1)
    delay_low = low.shift(1)
    delay_close = close.shift(1)

    hd = high - delay_high
    ld = delay_low - low
    tr = np.maximum(np.maximum(high - low, (delay_close - high).abs()), (delay_close - low).abs())

    ld_filtered = pd.DataFrame(np.where((ld > 0) & (ld > hd), ld, 0.0), index=close.index, columns=close.columns)
    hd_filtered = pd.DataFrame(np.where((hd > 0) & (hd > ld), hd, 0.0), index=close.index, columns=close.columns)

    sum_tr14 = tr.rolling(14).sum()
    minus_di = ld_filtered.rolling(14).sum() * 100 / sum_tr14
    plus_di = hd_filtered.rolling(14).sum() * 100 / sum_tr14

    dx = (minus_di - plus_di).abs() / (minus_di + plus_di) * 100
    alpha = dx.rolling(6).mean()
    return alpha


def alpha173(close):
    """
    Alpha#173: 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    """
    sma1 = close.ewm(alpha=2 / 13, adjust=False).mean()
    sma2 = sma1.ewm(alpha=2 / 13, adjust=False).mean()
    log_close = np.log(close)
    log_sma1 = log_close.ewm(alpha=2 / 13, adjust=False).mean()
    log_sma2 = log_sma1.ewm(alpha=2 / 13, adjust=False).mean()
    log_sma3 = log_sma2.ewm(alpha=2 / 13, adjust=False).mean()
    alpha = 3 * sma1 - 2 * sma2 + log_sma3
    return alpha


def alpha174(close):
    """
    Alpha#174: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    """
    std20 = close.rolling(20).std()
    delay_close = close.shift(1)
    inner = pd.DataFrame(np.where(close > delay_close, std20, 0.0), index=close.index, columns=close.columns)
    alpha = inner.ewm(alpha=1 / 20, adjust=False).mean()
    return alpha


def alpha175(close, high, low):
    """
    Alpha#175: 同 alpha161 逻辑，窗口改为6
    """
    delay_close = close.shift(1)
    tr = np.maximum(np.maximum(high - low, (delay_close - high).abs()), (delay_close - low).abs())
    alpha = tr.rolling(6).mean()
    return alpha


def alpha176(close, high, low, volume):
    """
    Alpha#176: CORR(RANK(((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12)))),RANK(VOLUME),6)
    """
    rsv = (close - low.rolling(12).min()) / (high.rolling(12).max() - low.rolling(12).min())
    rank_rsv = rsv.rank(axis=1, pct=True)
    rank_vol = volume.rank(axis=1, pct=True)
    alpha = rank_rsv.rolling(6).corr(rank_vol)
    return alpha


def alpha177(high):
    """
    Alpha#177: ((20-HIGHDAY(HIGH,20))/20)*100
    """
    highday = _highday(high, 20)
    alpha = (20 - highday) / 20 * 100
    return alpha


def alpha178(close, volume):
    """
    Alpha#178: (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    """
    alpha = (close - close.shift(1)) / close.shift(1) * volume
    return alpha


def alpha179(low, vwap, volume):
    """
    Alpha#179: (RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))
    """
    part1 = vwap.rolling(4).corr(volume).rank(axis=1, pct=True)
    rank_low = low.rank(axis=1, pct=True)
    rank_vol_mean50 = volume.rolling(50).mean().rank(axis=1, pct=True)
    part2 = rank_low.rolling(12).corr(rank_vol_mean50).rank(axis=1, pct=True)
    alpha = part1 * part2
    return alpha


def alpha180(close, volume):
    """
    Alpha#180: ((MEAN(VOLUME,20)<VOLUME) ? ((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))) : (-1*VOLUME))
    """
    vol_mean20 = volume.rolling(20).mean()
    delta7 = close.diff(7)
    active_signal = -1 * delta7.abs().rolling(60).rank(pct=True) * np.sign(delta7)
    alpha = pd.DataFrame(
        np.where(vol_mean20 < volume, active_signal, -1 * volume),
        index=close.index, columns=close.columns
    )
    return alpha


def alpha181(benchmarkindex, close):
    """
    Alpha#181: SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(benchmarkindexCLOSE-MEAN(benchmarkindexCLOSE,20))^2,20)/SUM((benchmarkindexCLOSE-MEAN(benchmarkindexCLOSE,20))^3,20)
    """
    bench_close = benchmarkindex["close"].reindex(close.index)
    stock_ret = close / close.shift(1) - 1
    stock_ret_deviation = stock_ret - stock_ret.rolling(20).mean()
    bench_deviation = bench_close - bench_close.rolling(20).mean()
    numerator = stock_ret_deviation.sub(bench_deviation.pow(2), axis=0).rolling(20).sum()
    denominator = bench_deviation.pow(3).rolling(20).sum()
    alpha = numerator.div(denominator.replace(0, np.nan), axis=0)
    return alpha


def alpha182(benchmarkindex, close, open):
    """
    Alpha#182: COUNT((CLOSE>OPEN & benchmarkindexCLOSE>benchmarkindexOPEN) OR (CLOSE<OPEN & benchmarkindexCLOSE<benchmarkindexOPEN),20)/20
    """
    bench_close = benchmarkindex["close"].reindex(close.index)
    bench_open = benchmarkindex["open"].reindex(close.index)
    stock_up = close > open
    stock_down = close < open
    bench_up = bench_close > bench_open
    bench_down = bench_close < bench_open
    same_up = stock_up.mul(bench_up, axis=0)
    same_down = stock_down.mul(bench_down, axis=0)
    same_direction = (same_up | same_down).astype(float)
    alpha = same_direction.rolling(20).sum() / 20
    return alpha


def alpha183(close):
    """
    Alpha#183: 与 alpha165 同一个 R/S(重标极差)统计量结构，窗口改为24
    """
    window = 24

    def rs_stat(x):
        if np.isnan(x).any():
            return np.nan
        dev = x - x.mean()
        cum = np.cumsum(dev)
        std = x.std()
        if std == 0:
            return np.nan
        return (cum.max() - cum.min()) / std

    alpha = close.rolling(window).apply(rs_stat, raw=True)
    return alpha


def alpha184(open, close):
    """
    Alpha#184: (RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200)) + RANK((OPEN-CLOSE)))
    """
    part1 = (open - close).shift(1).rolling(200).corr(close).rank(axis=1, pct=True)
    part2 = (open - close).rank(axis=1, pct=True)
    alpha = part1 + part2
    return alpha


def alpha185(open, close):
    """
    Alpha#185: RANK((-1*((1-(OPEN/CLOSE))^2)))
    """
    alpha = (-1 * ((1 - (open / close)) ** 2)).rank(axis=1, pct=True)
    return alpha


def alpha186(close, high, low):
    """
    Alpha#186: (ADX + DELAY(ADX,6))/2，ADX 即 alpha172 的 ADX 结构
    """
    adx = alpha172(close, high, low)
    alpha = (adx + adx.shift(6)) / 2
    return alpha


def alpha187(open, high, low):
    """
    Alpha#187: SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    这就是 alpha069/093 里 DTM（基于OPEN的标准定义）的20日滚动求和，直接复用 _dtm_dbm_open
    """
    dtm, _ = _dtm_dbm_open(open, high, low)
    alpha = dtm.rolling(20).sum()
    return alpha


def alpha188(high, low):
    """
    Alpha#188: ((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    原文的"–"是长破折号，等同减号
    """
    diff = high - low
    sma = diff.ewm(alpha=2 / 11, adjust=False).mean()
    alpha = (diff - sma) / sma * 100
    return alpha


def alpha189(close):
    """
    Alpha#189: MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    """
    mean6 = close.rolling(6).mean()
    alpha = (close - mean6).abs().rolling(6).mean()
    return alpha


def alpha190(close):
    """
    Alpha#190: LOG(((COUNT(r>g,20)-1)*SUMIF((r-g)^2,20,r<g)) / (COUNT(r<g,20)*SUMIF((r-g)^2,20,r>g)))
    其中 r=CLOSE/DELAY(CLOSE,1)-1（单日收益率），g=(CLOSE/DELAY(CLOSE,19))^(1/20)-1（20日几何平均日收益率基准）。
    本质是比较"高于/低于几何均值基准"两类交易日的离散度不对称性，取对数。
    原文里 "-1" 只出现在第一个 COUNT 项上（不对称），按字面保留，未做"修正对齐"。
    """
    r = close / close.shift(1) - 1
    g = (close / close.shift(19)) ** (1 / 20) - 1
    dev_sq = (r - g) ** 2

    cond_above = r > g
    cond_below = r < g

    count_above = cond_above.astype(float).rolling(20).sum()
    count_below = cond_below.astype(float).rolling(20).sum()

    sumif_below = pd.DataFrame(np.where(cond_below, dev_sq, 0.0), index=close.index, columns=close.columns).rolling(20).sum()
    sumif_above = pd.DataFrame(np.where(cond_above, dev_sq, 0.0), index=close.index, columns=close.columns).rolling(20).sum()

    numerator = (count_above - 1) * sumif_below
    denominator = count_below * sumif_above

    alpha = np.log(numerator / denominator)
    return alpha


def alpha191(close, high, low, volume):
    """
    Alpha#191: ((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)
    """
    corr = volume.rolling(20).mean().rolling(5).corr(low)
    alpha = (corr + (high + low) / 2) - close
    return alpha
