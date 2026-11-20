from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="Market Pattern Lab", page_icon="📈", layout="wide")
st.title("📈 Market Pattern Lab")
st.caption("OHLCV CSV 차트 · 캔들/차트 패턴 · 과거 빈도 · 몬테카를로 시나리오")

PATTERN_META = {
    "doji": ("도지", 0, "diamond", "#b0bec5"),
    "shooting_star": ("슈팅스타", -1, "triangle-down", "#ef5350"),
    "hammer": ("망치형", 1, "triangle-up", "#26a69a"),
    "bullish_engulfing": ("상승 장악형", 1, "star", "#00c853"),
    "bearish_engulfing": ("하락 장악형", -1, "star", "#d50000"),
    "ascending_triangle": ("상승 삼각형", 1, "triangle-up", "#42a5f5"),
    "rising_wedge": ("상승 쐐기", -1, "triangle-down", "#ff7043"),
    "falling_wedge": ("하락 쐐기", 1, "triangle-up", "#66bb6a"),
    "bear_pennant": ("베어 페넌트", -1, "triangle-down", "#ab47bc"),
    "head_shoulders": ("헤드앤숄더", -1, "x", "#e53935"),
    "inverse_head_shoulders": ("역헤드앤숄더", 1, "x", "#43a047"),
}


def guess_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower().replace(" ", "").replace("_", ""): str(c) for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    return None


def parse_time_column(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95:
        median = float(numeric.dropna().abs().median())
        unit = "ns" if median >= 1e17 else "us" if median >= 1e14 else "ms" if median >= 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(values, errors="coerce")


def normalize_ohlcv(raw: pd.DataFrame, columns: dict[str, str | None]) -> pd.DataFrame:
    required = ("open", "high", "low", "close")
    missing = [key for key in required if not columns.get(key)]
    if missing:
        raise ValueError("필수 열을 찾지 못했습니다: " + ", ".join(missing))

    data = pd.DataFrame()
    time_column = columns.get("time")
    data["time"] = parse_time_column(raw[time_column]) if time_column else pd.RangeIndex(len(raw))
    for key in required:
        data[key] = pd.to_numeric(raw[columns[key]], errors="coerce")
    volume_column = columns.get("volume")
    data["volume"] = pd.to_numeric(raw[volume_column], errors="coerce").fillna(0) if volume_column else 0.0

    data = data.dropna(subset=["time", *required]).copy()
    data = data[(data["open"] > 0) & (data["high"] > 0) & (data["low"] > 0) & (data["close"] > 0)]
    invalid_range = (
        (data["high"] < data[["open", "close", "low"]].max(axis=1))
        | (data["low"] > data[["open", "close", "high"]].min(axis=1))
        | (data["volume"] < 0)
    )
    data = data.loc[~invalid_range].sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if len(data) < 40:
        raise ValueError(f"정상 OHLCV 행이 {len(data)}개입니다. 패턴 계산에는 최소 40개가 필요합니다.")
    return data


def local_swings(highs: np.ndarray, lows: np.ndarray, left: int = 2, right: int = 2) -> list[tuple[int, str, float]]:
    points: list[tuple[int, str, float]] = []
    n = len(highs)
    for i in range(left, n - right):
        is_high = highs[i] > np.max(highs[i - left : i]) and highs[i] >= np.max(highs[i + 1 : i + right + 1])
        is_low = lows[i] < np.min(lows[i - left : i]) and lows[i] <= np.min(lows[i + 1 : i + right + 1])
        if is_high:
            points.append((i, "H", float(highs[i])))
        if is_low:
            points.append((i, "L", float(lows[i])))
    points.sort(key=lambda item: item[0])

    alternating: list[tuple[int, str, float]] = []
    for point in points:
        if not alternating or alternating[-1][1] != point[1]:
            alternating.append(point)
        elif (point[1] == "H" and point[2] > alternating[-1][2]) or (point[1] == "L" and point[2] < alternating[-1][2]):
            alternating[-1] = point
    return alternating


def _fit(points: list[tuple[int, str, float]]) -> tuple[float, float]:
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[2] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def detect_patterns(data: pd.DataFrame) -> pd.DataFrame:
    n = len(data)
    out = pd.DataFrame(False, index=data.index, columns=PATTERN_META.keys())
    o = data["open"].to_numpy(dtype=float)
    h = data["high"].to_numpy(dtype=float)
    l = data["low"].to_numpy(dtype=float)
    c = data["close"].to_numpy(dtype=float)

    candle_range = np.maximum(h - l, np.maximum(np.abs(c) * 1e-10, 1e-12))
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    body_ratio = body / candle_range

    out["doji"] = body_ratio <= 0.10
    out["shooting_star"] = (
        (upper_wick >= np.maximum(body * 2.0, candle_range * 0.35))
        & (lower_wick <= candle_range * 0.25)
        & (body_ratio <= 0.35)
        & ((np.minimum(o, c) - l) / candle_range <= 0.55)
    )
    out["hammer"] = (
        (lower_wick >= np.maximum(body * 2.0, candle_range * 0.35))
        & (upper_wick <= candle_range * 0.25)
        & (body_ratio <= 0.35)
        & ((h - np.maximum(o, c)) / candle_range <= 0.55)
    )
    open_s, close_s = data["open"], data["close"]
    out["bullish_engulfing"] = (
        (close_s > open_s) & (close_s.shift(1) < open_s.shift(1))
        & (open_s <= close_s.shift(1)) & (close_s >= open_s.shift(1))
    ).fillna(False)
    out["bearish_engulfing"] = (
        (close_s < open_s) & (close_s.shift(1) > open_s.shift(1))
        & (open_s >= close_s.shift(1)) & (close_s <= open_s.shift(1))
    ).fillna(False)

    active_keys = [
        "ascending_triangle", "rising_wedge", "falling_wedge",
        "bear_pennant", "head_shoulders", "inverse_head_shoulders",
    ]
    active = {key: False for key in active_keys}

    for i in range(35, n):
        start = max(0, i - 79)
        wh, wl, wc = h[start : i + 1], l[start : i + 1], c[start : i + 1]
        swings = local_swings(wh, wl, left=2, right=2)
        highs = [p for p in swings if p[1] == "H"][-5:]
        lows = [p for p in swings if p[1] == "L"][-5:]
        current = {key: False for key in active_keys}

        if len(highs) >= 2 and len(lows) >= 2:
            top_points, bottom_points = highs[-3:], lows[-3:]
            top_values = np.asarray([p[2] for p in top_points])
            top_mean = float(np.mean(top_values))
            resistance_flat = (float(np.max(top_values) - np.min(top_values)) / max(top_mean, 1e-12)) <= 0.018
            low_slope, low_intercept = _fit(bottom_points)
            support_now = low_slope * (len(wc) - 1) + low_intercept
            current["ascending_triangle"] = (
                resistance_flat and low_slope > 0
                and wc[-1] <= top_mean * 1.025 and wc[-1] >= support_now * 0.985
            )

            high_slope, high_intercept = _fit(top_points)
            x0 = max(min(p[0] for p in top_points), min(p[0] for p in bottom_points))
            x1 = len(wc) - 1
            width_start = (high_slope * x0 + high_intercept) - (low_slope * x0 + low_intercept)
            width_end = (high_slope * x1 + high_intercept) - (low_slope * x1 + low_intercept)
            narrowing = width_start > 0 and width_end > 0 and width_end < width_start * 0.82 and (x1 - x0) >= 8
            scale = max(float(np.median(wc)), 1e-12)
            if narrowing and high_slope > 0 and low_slope > high_slope and low_slope * (x1 - x0) / scale > 0.003:
                current["rising_wedge"] = True
            if narrowing and low_slope < 0 and high_slope > low_slope and high_slope * (x1 - x0) / scale < -0.003:
                current["falling_wedge"] = True

        if i >= 30:
            impulse_start, impulse_end = max(0, i - 30), max(1, i - 12)
            impulse_return = c[impulse_end] / max(c[impulse_start], 1e-12) - 1.0
            consolidation_start = max(0, i - 11)
            cons_h, cons_l = h[consolidation_start : i + 1], l[consolidation_start : i + 1]
            if len(cons_h) >= 8:
                x = np.arange(len(cons_h), dtype=float)
                hs, ls = float(np.polyfit(x, cons_h, 1)[0]), float(np.polyfit(x, cons_l, 1)[0])
                early_width, late_width = float(np.mean(cons_h[:4] - cons_l[:4])), float(np.mean(cons_h[-4:] - cons_l[-4:]))
                current["bear_pennant"] = impulse_return <= -0.05 and hs < 0 and ls > 0 and late_width < early_width * 0.80

        recent = swings[-5:]
        if len(recent) == 5:
            types = "".join(p[1] for p in recent)
            if types == "HLHLH":
                left_shoulder, _, head, _, right_shoulder = recent
                shoulder_mean = (left_shoulder[2] + right_shoulder[2]) / 2.0
                current["head_shoulders"] = (
                    shoulder_mean > 0 and abs(left_shoulder[2] - right_shoulder[2]) / shoulder_mean <= 0.05
                    and head[2] >= shoulder_mean * 1.025
                )
            elif types == "LHLHL":
                left_shoulder, _, head, _, right_shoulder = recent
                shoulder_mean = (left_shoulder[2] + right_shoulder[2]) / 2.0
                current["inverse_head_shoulders"] = (
                    shoulder_mean > 0 and abs(left_shoulder[2] - right_shoulder[2]) / shoulder_mean <= 0.05
                    and head[2] <= shoulder_mean * 0.975
                )

        for key in active_keys:
            out.at[i, key] = current[key] and not active[key]
            active[key] = current[key]

    return out.fillna(False).astype(bool)


def profile_levels(data: pd.DataFrame, bins: int = 32, value_area: float = 0.70) -> dict[str, float] | None:
    sample = data.tail(500)
    low, high = float(sample["low"].min()), float(sample["high"].max())
    if high <= low:
        return None
    typical = ((sample["high"] + sample["low"] + sample["close"]) / 3.0).to_numpy()
    volume = sample["volume"].to_numpy(dtype=float)
    if volume.sum() <= 0:
        return None
    hist, edges = np.histogram(typical, bins=bins, range=(low, high), weights=volume)
    poc = int(np.argmax(hist))
    left = right = poc
    target, area = float(hist.sum()) * value_area, float(hist[poc])
    while area < target and (left > 0 or right < bins - 1):
        left_v = hist[left - 1] if left > 0 else -1
        right_v = hist[right + 1] if right < bins - 1 else -1
        if right_v > left_v:
            right += 1
            area += max(float(right_v), 0.0)
        else:
            left -= 1
            area += max(float(left_v), 0.0)
    return {"POC": float((edges[poc] + edges[poc + 1]) / 2),
            "VAH": float(edges[right + 1]), "VAL": float(edges[left])}


def fib_levels(data: pd.DataFrame) -> dict[str, float] | None:
    sample = data.tail(250)
    swings = local_swings(sample["high"].to_numpy(dtype=float), sample["low"].to_numpy(dtype=float), left=3, right=3)
    if len(swings) < 2:
        return None
    first, last = swings[-2], swings[-1]
    low = first[2] if first[1] == "L" else last[2]
    high = first[2] if first[1] == "H" else last[2]
    if high <= low:
        return None
    ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    if first[1] == "L" and last[1] == "H":
        return {f"Fib {r:.3f}": high - (high - low) * r for r in ratios}
    return {f"Fib {r:.3f}": low + (high - low) * r for r in ratios}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def pattern_statistics(data: pd.DataFrame, signals: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    close = data["close"].to_numpy(dtype=float)
    last_complete = len(data) - horizon - 1
    for key, (label, direction, _, _) in PATTERN_META.items():
        events = np.flatnonzero(signals[key].to_numpy())
        events = events[events <= last_complete]
        if len(events):
            future_returns = close[events + horizon] / close[events] - 1.0
            if direction == 0:
                wins, title = int(np.sum(future_returns > 0)), "상승 빈도"
            else:
                wins, title = int(np.sum(future_returns * direction > 0), "방향 적중률")
            n = int(len(future_returns))
            rate = wins / n
            lower, upper = wilson_interval(wins, n)
            median_return = float(np.median(future_returns))
        else:
            n, wins, rate, lower, upper, median_return, title = 0, 0, np.nan, np.nan, np.nan, np.nan, "방향 적중률"
        rows.append({"key": key, "패턴": label, "표본": n, "성공": wins, "빈도": rate,
                     "95% 하한": lower, "95% 상한": upper, "중앙 수익률": median_return, "기준": title})
    return pd.DataFrame(rows)


def simulate_paths(data: pd.DataFrame, horizon: int, path_count: int, seed: int) -> np.ndarray:
    close = data["close"].to_numpy(dtype=float)
    log_returns = np.diff(np.log(close))
    sample = log_returns[-min(500, len(log_returns)) :]
    sample = sample[np.isfinite(sample)]
    if len(sample) < 10:
        raise ValueError("몬테카를로 계산에 필요한 유효 수익률이 부족합니다.")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(sample, size=(path_count, horizon), replace=True)
    return np.column_stack([np.full(path_count, close[-1]), close[-1] * np.exp(np.cumsum(sampled, axis=1))])


def make_price_chart(data: pd.DataFrame, signals: pd.DataFrame, profile: dict[str, float] | None,
                     fibs: dict[str, float] | None, visible_bars: int) -> go.Figure:
    view = data.tail(visible_bars)
    flags = signals.loc[view.index]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.78, 0.22], specs=[[{}], [{}]])
    fig.add_trace(go.Candlestick(x=view["time"], open=view["open"], high=view["high"],
                                 low=view["low"], close=view["close"], name="가격",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Bar(x=view["time"], y=view["volume"],
                         marker_color=np.where(view["close"] >= view["open"], "#26a69a", "#ef5350"),
                         name="거래량", opacity=0.65), row=2, col=1)

    for key, (label, direction, symbol, color) in PATTERN_META.items():
        hit = flags[key]
        if not hit.any():
            continue
        points = view.loc[hit]
        y = points["low"] * 0.995 if direction >= 0 else points["high"] * 1.005
        fig.add_trace(go.Scatter(
            x=points["time"], y=y, mode="markers", name=label,
            marker={"symbol": symbol, "size": 10, "color": color, "line": {"width": 1, "color": "white"}},
            customdata=np.column_stack([points["close"], points["volume"]]),
            hovertemplate=f"{label}<br>시간: %{{x}}<br>가격: %{{customdata[0]:,.6g}}<br>거래량: %{{customdata[1]:,.4g}}<extra></extra>",
        ), row=1, col=1)

    if profile:
        for name, color in (("POC", "#ffd54f"), ("VAH", "#42a5f5"), ("VAL", "#42a5f5")):
            fig.add_hline(y=profile[name], line_dash="solid" if name == "POC" else "dot",
                          line_color=color, annotation_text=f"{name} {profile[name]:,.6g}", row=1, col=1)
    if fibs:
        for name, price in fibs.items():
            fig.add_hline(y=price, line_dash="dash", line_color="rgba(171, 71, 188, 0.45)",
                          annotation_text=f"{name} {price:,.6g}", annotation_position="left", row=1, col=1)

    fig.update_layout(height=760, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      margin={"l": 20, "r": 20, "t": 35, "b": 10}, legend={"orientation": "h", "y": 1.02},
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(title_text="가격", row=1, col=1, side="right")
    fig.update_yaxes(title_text="거래량", row=2, col=1, side="right")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig


def make_monte_carlo_chart(paths: np.ndarray, last_price: float, horizon: int) -> go.Figure:
    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(horizon + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=q95, line={"color": "rgba(66,165,245,0.35)"}, name="95% 분위"))
    fig.add_trace(go.Scatter(x=x, y=q05, line={"color": "rgba(66,165,245,0.35)"}, fill="tonexty",
                             fillcolor="rgba(66,165,245,0.08)", name="5% 분위"))
    fig.add_trace(go.Scatter(x=x, y=q75, line={"color": "rgba(255,167,38,0.65)"}, name="75% 분위"))
    fig.add_trace(go.Scatter(x=x, y=q25, line={"color": "rgba(255,167,38,0.65)"}, fill="tonexty",
                             fillcolor="rgba(255,167,38,0.10)", name="25% 분위"))
    fig.add_trace(go.Scatter(x=x, y=q50, line={"color": "#ffffff", "width": 2}, name="중앙값"))
    fig.add_hline(y=last_price, line_dash="dot", line_color="gray", annotation_text=f"현재 {last_price:,.6g}")
    fig.update_layout(title=f"몬테카를로 부트스트랩 · {horizon}봉 시나리오", xaxis_title="미래 봉 수",
                      yaxis_title="가격 시나리오", height=440, template="plotly_dark",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      margin={"l": 20, "r": 20, "t": 50, "b": 20}, legend={"orientation": "h"})
    return fig


with st.sidebar:
    st.header("데이터와 설정")
    uploaded = st.file_uploader("OHLCV CSV 업로드", type=["csv"])
    st.caption("TradingView에서 내보낸 CSV 또는 시간, 시가, 고가, 저가, 종가, 거래량 열이 있는 CSV를 올리세요.")
    horizon = st.number_input("패턴 평가 기간 (봉)", min_value=1, max_value=250, value=20)
    visible_bars = st.slider("차트에 표시할 최근 봉", min_value=100, max_value=3000, value=600, step=100)
    profile_bins = st.slider("근사 매물대 구간 수", min_value=12, max_value=80, value=32, step=4)
    path_count = st.slider("몬테카를로 경로 수", min_value=250, max_value=5000, value=1500, step=250)
    mc_horizon = st.slider("몬테카를로 미래 봉 수", min_value=5, max_value=180, value=30, step=5)
    seed = st.number_input("시뮬레이션 시드", min_value=0, max_value=999999, value=42)

if uploaded is None:
    st.info("왼쪽에서 가격표 CSV를 올리면 차트와 분석이 표시됩니다.")
    st.markdown("**필수 열:** 시간(선택), 시가(Open), 고가(High), 저가(Low), 종가(Close)  \n**선택 열:** 거래량(Volume)")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"CSV를 읽지 못했습니다: {exc}")
    st.stop()
if raw.empty:
    st.error("CSV에 데이터가 없습니다.")
    st.stop()

with st.sidebar:
    st.subheader("열 연결")
    columns = [str(col) for col in raw.columns]
    defaults = {
        "time": guess_column(columns, ("time", "timestamp", "datetime", "date", "시간", "날짜")),
        "open": guess_column(columns, ("open", "시가")),
        "high": guess_column(columns, ("high", "고가")),
        "low": guess_column(columns, ("low", "저가")),
        "close": guess_column(columns, ("close", "종가")),
        "volume": guess_column(columns, ("volume", "vol", "거래량")),
    }
    time_default = defaults["time"]
    time_choice = st.selectbox("시간 열", ["(행 순서 사용)"] + columns,
                               index=columns.index(time_default) + 1 if time_default in columns else 0)
    mapping: dict[str, str | None] = {"time": None if time_choice == "(행 순서 사용)" else time_choice}
    for key, label in (("open", "시가"), ("high", "고가"), ("low", "저가"), ("close", "종가")):
        choices = ["(선택)"] + columns
        default = defaults[key]
        mapping[key] = st.selectbox(label, choices, index=columns.index(default) + 1 if default in columns else 0)
        if mapping[key] == "(선택)":
            mapping[key] = None
    volume_choices = ["(없음)"] + columns
    volume_default = defaults["volume"]
    mapping["volume"] = st.selectbox(
        "거래량", volume_choices, index=columns.index(volume_default) + 1 if volume_default in columns else 0
    )
    if mapping["volume"] == "(없음)":
        mapping["volume"] = None

try:
    df = normalize_ohlcv(raw, mapping)
except Exception as exc:
    st.error(str(exc))
    st.stop()

signals = detect_patterns(df)
profile = profile_levels(df, bins=profile_bins)
fibs = fib_levels(df)
stats = pattern_statistics(df, signals, int(horizon))
latest = df.iloc[-1]
current_price = float(latest["close"])
returns = df["close"].pct_change().dropna()
change_1 = float(returns.iloc[-1] * 100) if len(returns) else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 종가", f"{current_price:,.6g}")
m2.metric("최근 봉 변동", f"{change_1:+.2f}%")
m3.metric("데이터 행", f"{len(df):,}")
m4.metric("기간", f"{df['time'].iloc[0]} — {df['time'].iloc[-1]}")

st.subheader("가격 차트와 자동 패턴")
st.plotly_chart(make_price_chart(df, signals, profile, fibs, int(visible_bars)), use_container_width=True)

event_rows: list[dict[str, Any]] = []
for key, (label, direction, _, _) in PATTERN_META.items():
    for idx in np.flatnonzero(signals[key].to_numpy()):
        event_rows.append({
            "시간": df.at[idx, "time"], "패턴": label,
            "방향": "상승" if direction > 0 else "하락" if direction < 0 else "중립",
            "종가": float(df.at[idx, "close"]), "최근 이벤트": len(df) - 1 - int(idx),
        })
events = pd.DataFrame(event_rows)
if not events.empty:
    events = events.sort_values("시간", ascending=False)
    recent_events = events[events["최근 이벤트"] <= 100].head(40)
else:
    recent_events = pd.DataFrame(columns=["시간", "패턴", "방향", "종가", "최근 이벤트"])

left, right = st.columns([1.1, 1])
with left:
    st.subheader(f"패턴별 과거 빈도 · 다음 {int(horizon)}봉")
    shown = stats.copy()
    shown["빈도"] = shown.apply(lambda row: "표본 없음" if row["표본"] == 0 else f"{row['빈도'] * 100:.1f}%", axis=1)
    shown["95% 구간"] = shown.apply(
        lambda row: "—" if row["표본"] == 0 else f"{row['95% 하한'] * 100:.1f}%–{row['95% 상한'] * 100:.1f}%",
        axis=1,
    )
    shown["중앙 수익률"] = shown["중앙 수익률"].apply(lambda x: "—" if pd.isna(x) else f"{x * 100:+.2f}%")
    st.dataframe(shown[["패턴", "기준", "표본", "빈도", "95% 구간", "중앙 수익률"]],
                 hide_index=True, use_container_width=True)
    st.caption("업로드한 가격표에서 과거에 발생했고 평가 기간이 지난 신호만 계산합니다. 도지는 상승 빈도로 표시합니다.")
with right:
    st.subheader("최근 감지 이벤트")
    if recent_events.empty:
        st.write("최근 100봉 안에 감지된 패턴이 없습니다.")
    else:
        st.dataframe(recent_events[["시간", "패턴", "방향", "종가", "최근 이벤트"]],
                     hide_index=True, use_container_width=True)

st.subheader("몬테카를로 가격 시나리오")
try:
    paths = simulate_paths(df, int(mc_horizon), int(path_count), int(seed))
    terminal = paths[:, -1]
    above_rate = float(np.mean(terminal > current_price))
    q05, q50, q95 = np.percentile(terminal, [5, 50, 95])
    a, b, ccol = st.columns(3)
    a.metric(f"{mc_horizon}봉 후 중앙값", f"{q50:,.6g}")
    b.metric(f"{mc_horizon}봉 후 5–95% 범위", f"{q05:,.6g} — {q95:,.6g}")
    ccol.metric("현재가보다 높을 시나리오", f"{above_rate * 100:.1f}%")
    st.plotly_chart(make_monte_carlo_chart(paths, current_price, int(mc_horizon)), use_container_width=True)
except Exception as exc:
    st.warning(f"시뮬레이션을 계산하지 못했습니다: {exc}")

with st.expander("패턴 규칙과 해석 한계"):
    st.markdown(
        """
        - 캔들 패턴은 몸통·꼬리 비율을 사용한 규칙입니다: 도지, 슈팅스타, 망치형, 상승/하락 장악형.
        - 쐐기, 상승 삼각형, 베어 페넌트, 헤드앤숄더/역헤드앤숄더는 확인된 국소 고점·저점과 가격 기울기로 근사합니다. 교과서적 판정과 다를 수 있습니다.
        - 확률은 업로드 데이터 안에서 같은 규칙이 발생한 과거 사례의 방향 빈도입니다. 표본 수와 95% Wilson 구간을 함께 확인하세요. 적은 표본은 불안정합니다.
        - 몬테카를로는 최근 로그수익률을 복원추출하는 시나리오입니다. 미래 가격이나 수익을 보장하지 않으며, 거래 비용·슬리피지·유동성·시장 구조 변화는 반영하지 않습니다.
        - 매물대는 봉의 대표가격(HLC3)에 거래량을 배분한 근사 POC/VAH/VAL입니다. 실제 체결별 매수·매도 흐름은 아닙니다.
        """
    )
with st.expander("현재 업로드 데이터"):
    st.dataframe(df.tail(200), hide_index=True, use_container_width=True)
