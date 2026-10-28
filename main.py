"""
Gold (XAU) Signal Website - بدون MT5
التشغيل:
    pip install fastapi uvicorn yfinance pandas numpy
    uvicorn main:app --host 0.0.0.0 --port 8000
ثم افتح http://127.0.0.1:8000
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import time

app = FastAPI(title="Gold Signal")
TICKER = "GC=F"          # عقود الذهب العالمية (قريبة جداً من XAUUSD)
STOP_ATR, TARGET_R = 1.6, 2.0
_cache = {"t": 0, "data": None}


def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    out = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return out.mask((ad == 0) & (au > 0), 100)

def tr_of(df):
    p = df.close.shift(1)
    return pd.concat([df.high - df.low, (df.high - p).abs(), (df.low - p).abs()], axis=1).max(axis=1)

def atr(df, n=14): return tr_of(df).ewm(alpha=1/n, min_periods=n, adjust=False).mean()

def adx_di(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    s = tr_of(df).ewm(alpha=1/n, min_periods=n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / s
    mdi = 100 * mdm.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean(), pdi, mdi

def macd_hist(c):
    m = ema(c, 12) - ema(c, 26)
    return m - ema(m, 9)


def clean(raw):
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_convert("UTC") if idx.tz is not None else idx).tz_localize(None)
    df.index.name = "time"
    return df.reset_index()


def higher(df, prefix, delta):
    z = df.copy()
    z[prefix + "_up"] = ema(z.close, 20) > ema(z.close, 50)
    z[prefix + "_dn"] = ema(z.close, 20) < ema(z.close, 50)
    z[prefix + "_mh"] = macd_hist(z.close)
    z["time"] = z.time + delta  # تتاح البيانات بعد إغلاق الشمعة
    return z[["time", prefix + "_up", prefix + "_dn", prefix + "_mh"]]


def analyze():
    h1 = clean(yf.download(TICKER, period="90d", interval="1h", progress=False, auto_adjust=False))
    d1 = clean(yf.download(TICKER, period="1y", interval="1d", progress=False, auto_adjust=False))
    h1 = h1.iloc[:-1].reset_index(drop=True)  # حذف الشمعة الجارية
    if d1.time.iloc[-1].date() >= pd.Timestamp.utcnow().date():
        d1 = d1.iloc[:-1]
    h4 = (h1.set_index("time").resample("4h")
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna().reset_index().iloc[:-1])
    if len(h1) < 120:
        raise RuntimeError("البيانات غير كافية (ربما السوق مغلق أو Yahoo لا يستجيب)")

    x = h1.copy()
    x["ef"], x["es"] = ema(x.close, 20), ema(x.close, 50)
    x["slope"] = x.ef - x.ef.shift(3)
    x["rsi"], x["atr"] = rsi(x.close), atr(x)
    x["adx"], x["pdi"], x["mdi"] = adx_di(x)
    x["mid"] = x.close.rolling(20).mean()
    x["mh"] = macd_hist(x.close)
    x["ph"] = x.high.shift(1).rolling(20).max()
    x["pl"] = x.low.shift(1).rolling(20).min()
    x["volok"] = x.atr.div(x.close).between(0.0003, 0.007)
    x["notover"] = (x.close - x.ef).abs() < 2.5 * x.atr
    x = pd.merge_asof(x, higher(h4, "h4", pd.Timedelta(hours=4)), on="time")
    x = pd.merge_asof(x, higher(d1, "d1", pd.Timedelta(days=1)), on="time")
    L = x.iloc[-1]

    def f(v): return bool(v) if not pd.isna(v) else False
    buy = {
        "الاتجاه على H1 صاعد": f(L.ef > L.es and L.slope > 0),
        "H4 و D1 صاعدان": f(L.h4_up and L.d1_up),
        "MACD على H4 إيجابي": f(L.h4_mh > 0),
        "RSI بين 52 و70": f(52 <= L.rsi <= 70),
        "ADX ≥ 25 والمشترون أقوى": f(L.adx >= 25 and L.pdi > L.mdi),
        "السعر فوق متوسط بولنجر وMACD إيجابي": f(L.close > L.mid and L.mh > 0),
        "اختراق أعلى 20 شمعة": f(L.close > L.ph),
        "تذبذب مقبول وغير مبالغ فيه": f(L.volok and L.notover),
    }
    sell = {
        "الاتجاه على H1 هابط": f(L.ef < L.es and L.slope < 0),
        "H4 و D1 هابطان": f(L.h4_dn and L.d1_dn),
        "MACD على H4 سلبي": f(L.h4_mh < 0),
        "RSI بين 30 و48": f(30 <= L.rsi <= 48),
        "ADX ≥ 25 والبائعون أقوى": f(L.adx >= 25 and L.mdi > L.pdi),
        "السعر تحت متوسط بولنجر وMACD سلبي": f(L.close < L.mid and L.mh < 0),
        "كسر أدنى 20 شمعة": f(L.close < L.pl),
        "تذبذب مقبول وغير مبالغ فيه": f(L.volok and L.notover),
    }
    bs, ss = sum(buy.values()), sum(sell.values())
    a = float(L.atr)
    price = float(L.close)
    side, score, checks = "WAIT", max(bs, ss), buy if bs >= ss else sell
    if bs >= 6 and bs > ss: side, score, checks = "BUY", bs, buy
    elif ss >= 6 and ss > bs: side, score, checks = "SELL", ss, sell
    sl = tp = None
    if side == "BUY":
        sl, tp = price - STOP_ATR * a, price + STOP_ATR * a * TARGET_R
    elif side == "SELL":
        sl, tp = price + STOP_ATR * a, price - STOP_ATR * a * TARGET_R
    strength = "قوية" if score == 8 else ("متوسطة" if side != "WAIT" else "—")

    t = x.tail(150)
    ts = lambda s: int(s.timestamp())
    return {
        "side": side, "strength": strength, "score": score,
        "price": round(price, 2),
        "sl": round(sl, 2) if sl else None, "tp": round(tp, 2) if tp else None,
        "atr": round(a, 2), "adx": round(float(L.adx), 1), "rsi": round(float(L.rsi), 1),
        "time": str(L.time) + " UTC",
        "checks": [{"name": k, "ok": v} for k, v in checks.items()],
        "candles": [{"time": ts(r.time), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                    for r in t.itertuples()],
        "ema20": [{"time": ts(r.time), "value": r.ef} for r in t.itertuples()],
        "ema50": [{"time": ts(r.time), "value": r.es} for r in t.itertuples()],
    }


@app.get("/api/signal")
def api_signal():
    if time.time() - _cache["t"] > 55 or _cache["data"] is None:
        try:
            _cache["data"] = {"status": "ok", **analyze()}
            _cache["t"] = time.time()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return _cache["data"]


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


PAGE = """<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>توصيات الذهب</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lightweight-charts/4.1.3/lightweight-charts.standalone.production.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Tahoma,sans-serif;background:#0f172a;color:#f8fafc;padding:12px}
.box{background:#1e293b;border-radius:12px;padding:14px;margin-bottom:12px}
#sig{font-size:30px;font-weight:bold;text-align:center;padding:8px}
.BUY{color:#34d399}.SELL{color:#f87171}.WAIT{color:#fbbf24}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center}
.grid div{background:#0f172a;border-radius:8px;padding:10px}.grid small{color:#94a3b8;display:block}
.grid b{font-size:18px}#chart{height:360px}li{list-style:none;padding:3px 0;font-size:14px}
.sub{color:#94a3b8;font-size:12px;text-align:center}
</style></head><body>
<div class="box"><div class="sub">الذهب العالمي (GC=F) - فريم الساعة</div>
<div id="sig" class="WAIT">...جاري التحليل</div><div id="str" class="sub"></div></div>
<div class="box grid">
<div><small>الدخول</small><b id="en">-</b></div>
<div><small>وقف الخسارة</small><b id="sl" style="color:#f87171">-</b></div>
<div><small>الهدف</small><b id="tp" style="color:#34d399">-</b></div></div>
<div class="box"><div id="chart"></div></div>
<div class="box"><b>شروط الاتجاه الحالي</b><ul id="ck" style="padding:0;margin:8px 0 0"></ul>
<div class="sub" id="meta"></div></div>
<div class="sub">تحليل فني تعليمي وليس نصيحة مالية. جرّب على حساب تجريبي وضع وقف الخسارة دائماً.</div>
<script>
const chart=LightweightCharts.createChart(document.getElementById('chart'),{layout:{background:{color:'#1e293b'},textColor:'#cbd5e1'},grid:{vertLines:{color:'#273449'},horzLines:{color:'#273449'}},timeScale:{timeVisible:true},autoSize:true});
const cs=chart.addCandlestickSeries({upColor:'#34d399',downColor:'#f87171',borderVisible:false,wickUpColor:'#34d399',wickDownColor:'#f87171'});
const e20=chart.addLineSeries({color:'#60a5fa',lineWidth:1}),e50=chart.addLineSeries({color:'#fbbf24',lineWidth:1});
let lines=[];
const names={BUY:'شراء BUY',SELL:'بيع SELL',WAIT:'انتظار WAIT'};
async function load(){
 try{const d=await (await fetch('/api/signal')).json();
 if(d.status!=='ok'){document.getElementById('sig').textContent='خطأ: '+d.message;return}
 const s=document.getElementById('sig');s.className=d.side;s.textContent=names[d.side];
 document.getElementById('str').textContent=d.side=='WAIT'?'لا توجد فرصة واضحة الآن':'قوة الإشارة: '+d.strength+' ('+d.score+'/8)';
 document.getElementById('en').textContent=d.price;
 document.getElementById('sl').textContent=d.sl??'-';document.getElementById('tp').textContent=d.tp??'-';
 cs.setData(d.candles);e20.setData(d.ema20);e50.setData(d.ema50);
 lines.forEach(l=>cs.removePriceLine(l));lines=[];
 if(d.sl){[[d.price,'#60a5fa','دخول'],[d.sl,'#f87171','SL'],[d.tp,'#34d399','TP']].forEach(([p,c,t])=>lines.push(cs.createPriceLine({price:p,color:c,lineWidth:2,title:t})))}
 document.getElementById('ck').innerHTML=d.checks.map(c=>`<li>${c.ok?'✅':'❌'} ${c.name}</li>`).join('');
 document.getElementById('meta').textContent=`آخر شمعة: ${d.time} | ATR ${d.atr} | ADX ${d.adx} | RSI ${d.rsi}`;
 }catch(e){document.getElementById('sig').textContent='تعذر الاتصال بالسيرفر'}}
load();setInterval(load,60000);
</script></body></html>"""
