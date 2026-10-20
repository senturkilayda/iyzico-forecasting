"""
IYZICO - ÜYE İŞ YERİ BAZINDA 92 GÜNLÜK ÖDEME HACMİ TAHMİNİ
===========================================================

Akış:
  1. Veriyi yükle ve veri kalitesini raporla
  2. Eksiksiz günlük panel kur (iş yeri başlangıç tarihlerine saygılı)
  3. Takvim + tatil + lag + rolling + EWM özellikleri
  4. Baseline'lar ve LightGBM (iki hedef tanımı) - iki validation fold'u
  5. Final model -> Ekim-Aralık 2020 tahmini
  6. Yayın kalitesinde görseller (./figures) ve çıktılar (./outputs)

Çalıştırma:  python iyzico_forecasting.py
Gereken veri: ./data/iyzico_data.csv
Gerekli kolonlar: merchant_id, transaction_date, Total_Transaction, Total_Paid
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter
import lightgbm as lgb


###################################################
# AYARLAR
###################################################

# Veri dosyasının yolu. Otomatik bulunamazsa buraya TAM yolu yaz, örneğin:
# DATA_PATH_OVERRIDE = Path(r"C:\Users\KULLANICI\PyCharmMiscProject\data\iyzico_data.csv")
DATA_PATH_OVERRIDE = None
DATA_FILE_NAME = "iyzico_data.csv"
FIG_DIR = Path.cwd() / "figures"   # main() içinde proje klasörüne göre güncellenir
OUT_DIR = Path.cwd() / "outputs"
SHOW_PLOTS = True            # Sunucuda çalıştırıyorsan False yap

HORIZON = 92                 # Tahmin ufku (gün)
LAST_OBSERVED = pd.Timestamp("2020-09-30")   # Bilinen son gün
FORECAST_END = pd.Timestamp("2020-12-31")    # Tahmin edilecek son gün

# Her fold: (validation başlangıcı, validation bitişi)
# Train, validation başlangıcından ÖNCEKİ tüm günlerdir.
FOLDS = {
    "Fold A (Eki-Ara 2019)": ("2019-10-01", "2019-12-31"),
    "Fold B (Tem-Eyl 2020)": ("2020-07-01", "2020-09-30"),
}

LAG_DAYS = [92, 98, 182, 364, 366]
ROLL_WINDOWS = [7, 28, 91]
EWM_SPANS = [7, 28, 91]
SEED = 42

# COVID dönemi (grafiklerde gölge, bayrak ve çıkarma için ortak tanım).
# Kendi verindeki iş yerlerinin grafiklerine bakıp gerekirse güncelle.
COVID_START = pd.Timestamp("2020-03-11")
COVID_END = pd.Timestamp("2020-06-01")
# include : COVID günleri olduğu gibi modele girer
# flag    : include + "is_covid" bayrağı (test döneminde 0)
# exclude : COVID günlerinin değerleri tamamen NaN yapılır; hem eğitim
#           hedeflerinden hem lag/rolling/EWM kaynaklarından çıkar
COVID_MODES = ["include", "flag", "exclude"]
# Final seçimde: en iyi skordan bu kadar (SMAPE puanı) içindeki adaylar
# arasında en BASİT olanı seç. Farklar gürültü düzeyindeyse karmaşıklık
# (ör. ek tatil özellikleri) tercih edilmez.
SELECTION_TOLERANCE = 0.10

BASELINES = {
    "Lag 92": "paid_lag_92",
    "Lag 98": "paid_lag_98",
    "Lag 364": "paid_lag_364",
    "Lag 366": "paid_lag_366",
    "Rolling Mean 28": "paid_roll_mean_28",
    "Rolling Mean 91": "paid_roll_mean_91",
}

# Takvim bilgisi (elle girildi - kullanmadan önce bir kez doğrula).
# Dini bayramlar: (ilk gün, son gün)
BAYRAM_PERIODS = [
    ("2018-06-15", "2018-06-17"),  # Ramazan Bayramı 2018
    ("2018-08-21", "2018-08-24"),  # Kurban Bayramı 2018
    ("2019-06-04", "2019-06-06"),  # Ramazan Bayramı 2019
    ("2019-08-11", "2019-08-14"),  # Kurban Bayramı 2019
    ("2020-05-24", "2020-05-26"),  # Ramazan Bayramı 2020
    ("2020-07-31", "2020-08-03"),  # Kurban Bayramı 2020
]
FIXED_HOLIDAYS = {(1, 1), (4, 23), (5, 1), (5, 19), (7, 15), (8, 30), (10, 29)}
BLACK_FRIDAYS = ["2018-11-23", "2019-11-29", "2020-11-27"]
# Yarım günler: bayram arifeleri (bayramın ilk gününden bir önceki gün,
# BAYRAM_PERIODS'tan türetilir) + 28 Ekim (Cumhuriyet Bayramı arifesi).
HALF_DAYS_FIXED = {(10, 28)}
# Kapsam dışı: kamu kesimine verilen idari izinler (ör. 2018 Kurban arifesi)
# ve köprü günleri özel sektör ödeme davranışını etkileyebilir ama eklenmedi.


###################################################
# GÖRSEL STİL
###################################################

MERCHANT_COLORS = [
    "#2A6F97", "#C8553D", "#3A7D44", "#8E5572",
    "#D4A017", "#4E6E81", "#7B4B94",
]
YEAR_COLORS = {2018: "#A9C4D6", 2019: "#2A6F97", 2020: "#C8553D"}
INK = "#202124"
MUTED = "#5f6368"
GRID = "#e8eaed"
MONTH_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz",
            "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
WEEKDAY_TR = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9aa0a6",
        "axes.labelcolor": "#3c4043",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def finish_figure(fig, title, filename, note=None):
    """Başlık, dipnot ekler, kaydeder ve gösterir."""
    fig.suptitle(title, x=0.005, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    if note:
        fig.text(0.005, -0.012, note, ha="left", va="top",
                 fontsize=8, color=MUTED)
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    print("Kaydedildi:", FIG_DIR / filename)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def merchant_grid(n_merchants, height=3.0, sharey=False):
    """Her iş yeri için bir panel + (varsa) lejant için boş panel."""
    ncols = 2
    nrows = int(np.ceil((n_merchants + 1) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, height * nrows),
        layout="constrained", sharey=sharey
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_merchants:]:
        ax.axis("off")
    return fig, axes


def spare_legend(axes, n_merchants, handles, title=None):
    """Kullanılmayan paneli lejant alanı olarak kullanır."""
    ax = axes[n_merchants]
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", title=title,
              title_fontproperties={"weight": "bold"}, fontsize=10)


###################################################
# METRİKLER
###################################################

def smape(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    numerator = 2 * np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    errors = np.divide(numerator, denominator,
                       out=np.zeros_like(numerator),
                       where=denominator != 0)
    return 100 * np.mean(errors)


def wape(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    total_actual = np.abs(actual).sum()
    if total_actual == 0:
        return np.nan
    return 100 * np.abs(actual - predicted).sum() / total_actual


def mae(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(actual - predicted))


def evaluate(merchant, actual, predicted):
    """Genel ve iş yeri bazında ortalama (macro) metrikleri döndürür."""
    tmp = pd.DataFrame({"m": np.asarray(merchant),
                        "a": np.asarray(actual, dtype=float),
                        "p": np.asarray(predicted, dtype=float)})
    macro = np.mean([smape(g["a"], g["p"]) for _, g in tmp.groupby("m")])
    return {
        "Rows": len(tmp),
        "SMAPE (%)": smape(tmp["a"], tmp["p"]),
        "Macro SMAPE (%)": macro,
        "WAPE (%)": wape(tmp["a"], tmp["p"]),
        "MAE": mae(tmp["a"], tmp["p"]),
    }


###################################################
# 1. VERİ YÜKLEME VE KALİTE KONTROL
###################################################

def find_data_file():
    """
    Veri dosyasını sırayla şu yerlerde arar: manuel yol, çalışma klasörü,
    script'in klasörü, kullanıcı klasörü altındaki PyCharm projeleri.
    """
    if DATA_PATH_OVERRIDE is not None:
        return Path(DATA_PATH_OVERRIDE)

    name = DATA_FILE_NAME
    roots = [Path.cwd()]
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:          # PyCharm konsolu / interaktif ortam
        pass
    roots += [p for p in Path.home().glob("PyCharm*") if p.is_dir()]

    candidates = []
    for root in roots:
        candidates += [root / "data" / name, root / name]
    for cand in candidates:
        if cand.exists():
            return cand

    # Son çare: bulunan proje klasörlerinde alt klasörlere bak
    for root in roots[1:] if len(roots) > 1 else roots:
        for hit in root.rglob(name):
            if ".venv" not in hit.parts:
                return hit

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{name} bulunamadı. Aranan yerler:\n  {tried}\n"
        "DATA_PATH_OVERRIDE değişkenine dosyanın tam yolunu yaz."
    )


def load_data(path):
    print("Çalışma klasörü:", Path.cwd())
    print("Veri dosyası:", path)
    print("Dosya bulundu mu?", path.exists())

    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    required = {"merchant_id", "transaction_date",
                "Total_Transaction", "Total_Paid"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Eksik kolonlar: {missing_cols}")

    if df.duplicated(["merchant_id", "transaction_date"]).any():
        raise ValueError("Tekrarlanan iş yeri-tarih kayıtları var!")

    df = df.sort_values(["merchant_id", "transaction_date"])
    df = df.reset_index(drop=True)

    print("\nShape:", df.shape)
    print("Tarih aralığı:", df["transaction_date"].min().date(),
          "->", df["transaction_date"].max().date())
    print("İş yeri sayısı:", df["merchant_id"].nunique())
    return df


def split_off_future(df):
    """
    LAST_OBSERVED sonrasındaki satırları hedef bilinmiyormuş gibi maskeler.
    Özellikler en fazla t-92 günü kullandığı için bu güvenlidir.
    Sonrasında gerçek (pozitif) değer varsa test doğrulaması için saklanır.
    """
    post = df["transaction_date"] > LAST_OBSERVED
    truth = None

    if post.any():
        vals = df.loc[post, ["merchant_id", "transaction_date",
                             "Total_Paid"]].dropna()
        if (vals["Total_Paid"] > 0).any():
            truth = vals.set_index(["merchant_id", "transaction_date"])[
                "Total_Paid"]
            print(f"\nUYARI: {LAST_OBSERVED.date()} sonrasında "
                  f"{len(truth)} dolu satır var. Eğitime katılmayacak, "
                  "yalnızca test skoru için saklanacak.")
        df.loc[post, ["Total_Paid", "Total_Transaction"]] = np.nan

    if (df["Total_Paid"] < 0).any():
        raise ValueError("Negatif ödeme tutarı bulundu!")
    return df, truth


###################################################
# 2. PANEL
###################################################

def build_panel(df):
    """
    Her iş yeri için ilk gözlem gününden FORECAST_END'e kadar eksiksiz
    günlük takvim kurar. İş yerinin var olmadığı günler (ilk gözlemden
    öncesi) panele alınmaz; yaşam süresi içindeki eksik günler NaN kalır.
    """
    merchant_ids = sorted(df["merchant_id"].unique())
    all_dates = pd.date_range(df["transaction_date"].min(),
                              FORECAST_END, freq="D")
    full_index = pd.MultiIndex.from_product(
        [merchant_ids, all_dates],
        names=["merchant_id", "transaction_date"]
    )

    panel = (df.set_index(["merchant_id", "transaction_date"])
               .reindex(full_index)
               .reset_index())

    first_obs = (df.groupby("merchant_id")["transaction_date"].min()
                   .rename("first_obs"))
    panel = panel.merge(first_obs, left_on="merchant_id", right_index=True)
    panel = panel[panel["transaction_date"] >= panel["first_obs"]]
    panel = panel.drop(columns="first_obs")

    panel["is_observed"] = panel["Total_Paid"].notna().astype(int)
    panel = panel.sort_values(["merchant_id", "transaction_date"])
    return panel.reset_index(drop=True)


def data_quality_report(panel):
    obs = panel[panel["transaction_date"] <= LAST_OBSERVED]
    report = obs.groupby("merchant_id").agg(
        first_day=("transaction_date", "min"),
        days_in_life=("transaction_date", "count"),
        observed_days=("is_observed", "sum"),
    )
    report["last_observed_day"] = (
        obs[obs["is_observed"] == 1]
        .groupby("merchant_id")["transaction_date"].max()
    )
    report["missing_days"] = report["days_in_life"] - report["observed_days"]
    report["missing_pct"] = 100 * report["missing_days"] / report["days_in_life"]

    print("\nVERİ KALİTESİ (iş yeri yaşam süresi içinde)")
    print(report.to_string(float_format=lambda x: f"{x:,.2f}"))


###################################################
# 3. ÖZELLİKLER
###################################################

def add_calendar_features(data):
    d = data["transaction_date"]
    out = data.copy()

    out["month"] = d.dt.month
    out["day_of_month"] = d.dt.day
    out["day_of_year"] = d.dt.dayofyear
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["day_of_week"] = d.dt.dayofweek
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)

    # Dini bayram günleri
    bayram_days = set()
    for start, end in BAYRAM_PERIODS:
        bayram_days.update(pd.date_range(start, end))
    out["is_bayram"] = d.isin(bayram_days).astype(int)

    # Resmi tatiller (sabit tarihli + bayramlar)
    is_fixed = [(m, day) in FIXED_HOLIDAYS
                for m, day in zip(d.dt.month, d.dt.day)]
    out["is_holiday"] = (np.array(is_fixed) | (out["is_bayram"] == 1)).astype(int)

    # Yarım günler: bayram arifeleri + 28 Ekim
    arife_days = set(pd.to_datetime([s for s, _ in BAYRAM_PERIODS])
                     - pd.Timedelta(days=1))
    is_half_fixed = np.array([(m, day) in HALF_DAYS_FIXED
                              for m, day in zip(d.dt.month, d.dt.day)])
    out["is_half_day"] = (is_half_fixed
                          | d.isin(arife_days).to_numpy()).astype(int)

    # Bayrama kalan gün / bayramdan geçen gün (30 ile sınırlı)
    starts = pd.to_datetime([s for s, _ in BAYRAM_PERIODS])
    ends = pd.to_datetime([e for _, e in BAYRAM_PERIODS])
    to_next, since_last = {}, {}
    for date in pd.DatetimeIndex(d.unique()):
        ahead = np.asarray((starts - date).days)
        ahead = ahead[ahead >= 0]
        behind = np.asarray((date - ends).days)
        behind = behind[behind > 0]
        to_next[date] = min(ahead.min(), 30) if ahead.size else 30
        since_last[date] = min(behind.min(), 30) if behind.size else 30
    out["days_to_bayram"] = d.map(to_next).astype(int)
    out["days_since_bayram"] = d.map(since_last).astype(int)

    # Black Friday (±3 gün) ve 11.11
    bf = pd.to_datetime(BLACK_FRIDAYS)
    near_bf = np.zeros(len(out), dtype=bool)
    for day in bf:
        near_bf |= (np.abs((d - day).dt.days) <= 3).to_numpy()
    out["is_black_friday_window"] = near_bf.astype(int)
    out["is_1111"] = ((d.dt.month == 11) & (d.dt.day == 11)).astype(int)
    return out


BASE_CAL_COLS = [
    "month", "day_of_month", "day_of_year", "week_of_year", "day_of_week",
    "is_weekend", "is_month_start", "is_month_end",
]
HOLIDAY_COLS = [
    "is_holiday", "is_bayram", "is_half_day",
    "days_to_bayram", "days_since_bayram",
]
SHOPPING_COLS = ["is_black_friday_window", "is_1111"]
CALENDAR_COLS = BASE_CAL_COLS + HOLIDAY_COLS + SHOPPING_COLS

# Ablation: hangi özellik grubu gerçekten fayda sağlıyor?
FEATURE_SETS = {
    "tatil yok": [],
    "+tatil/bayram": HOLIDAY_COLS,
    "+tatil +BF/11.11": HOLIDAY_COLS + SHOPPING_COLS,
}


def add_history_features(panel):
    """
    Lag, rolling mean ve EWM özellikleri.
    Tüm özellikler t-92 veya daha eski günleri kullanır: tahmin başlangıcında
    gerçekten bilinen bilgi. Panel eksiksiz günlük olduğundan shift(n) = n gün.
    """
    data = panel.sort_values(["merchant_id", "transaction_date"])
    data = data.reset_index(drop=True)

    grouped = data.groupby("merchant_id")["Total_Paid"]
    new = {}

    for lag in LAG_DAYS:
        new[f"paid_lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(HORIZON).groupby(data["merchant_id"])

    for window in ROLL_WINDOWS:
        min_p = max(1, window // 2)
        new[f"paid_roll_mean_{window}"] = shifted.transform(
            lambda x, w=window, m=min_p: x.rolling(w, min_periods=m).mean()
        )

    for span in EWM_SPANS:
        min_p = max(1, span // 2)
        new[f"paid_ewm_{span}"] = shifted.transform(
            lambda x, s=span, m=min_p: x.ewm(
                span=s, adjust=False, min_periods=m).mean()
        )

    return pd.concat([data, pd.DataFrame(new)], axis=1)


LAG_COLS = [f"paid_lag_{lag}" for lag in LAG_DAYS]
ROLL_COLS = [f"paid_roll_mean_{w}" for w in ROLL_WINDOWS]
EWM_COLS = [f"paid_ewm_{s}" for s in EWM_SPANS]
FEATURE_COLS = CALENDAR_COLS + LAG_COLS + ROLL_COLS + EWM_COLS


def add_covid_flag(panel):
    out = panel.copy()
    out["is_covid"] = out["transaction_date"].between(
        COVID_START, COVID_END).astype(int)
    return out


def build_design_matrix(panel, covid_mode="include"):
    """Tüm panel için tek seferde One-Hot yaparak sütun uyumunu garanti eder."""
    cols = FEATURE_COLS + (["is_covid"] if covid_mode == "flag" else [])
    X = pd.get_dummies(
        panel[cols + ["merchant_id"]],
        columns=["merchant_id"], prefix="merchant", dtype=int
    )
    forbidden = {"Total_Paid", "Total_Transaction", "is_observed"}
    leaked = forbidden.intersection(X.columns)
    if leaked:
        raise ValueError(f"Hedef sızıntısı: {leaked}")
    return X


###################################################
# 4. MODEL
###################################################

def columns_for(X, feature_set):
    """Seçilen özellik setinde olmayan tatil/alışveriş sütunlarını çıkarır."""
    excluded = set(HOLIDAY_COLS + SHOPPING_COLS) - set(FEATURE_SETS[feature_set])
    return [c for c in X.columns if c not in excluded]


def make_model(n_estimators):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        random_state=SEED,
        verbosity=-1,
    )


def make_target(panel, level_log, mode):
    """
    plain : log1p(Total_Paid)
    level : log1p(Total_Paid) - log1p(91 günlük kaydırılmış ortalama)
            Büyüme/seviye kaymasını modelden çıkarır; ağaç modellerinin
            eğitim aralığının dışına çıkamama sorununu hafifletir.
    """
    y = np.log1p(panel["Total_Paid"])
    return y - level_log if mode == "level" else y


def to_paid_scale(pred_log, level_log, mode):
    pred = pred_log + level_log if mode == "level" else pred_log
    return np.clip(np.expm1(pred), 0, None)


MODEL_LABELS = {
    "plain": "LightGBM (log)",
    "level": "LightGBM (log, seviye normalize)",
}


def model_name(mode, feature_set, covid_mode):
    return f"{MODEL_LABELS[mode]} [{feature_set}] | COVID {covid_mode}"


def build_candidates():
    """Final seçime giren TÜM modeller: baseline'lar + LightGBM varyantları."""
    cands = [{"name": n, "kind": "baseline", "col": c}
             for n, c in BASELINES.items()]
    for covid in COVID_MODES:
        for mode in ("plain", "level"):
            for fs in FEATURE_SETS:
                cands.append({"name": model_name(mode, fs, covid),
                              "kind": "lgbm", "mode": mode, "fs": fs,
                              "covid": covid})
    return cands


def fold_common_masks(data):
    """
    Tüm adayların (tüm COVID modları dahil) aynı validation satırlarında
    değerlendirilmesi için ortak satır kümesi.
    """
    panel_inc = data["include"][0]
    masks = {}
    for fold_name, (vs, ve) in FOLDS.items():
        in_val = panel_inc["transaction_date"].between(
            pd.Timestamp(vs), pd.Timestamp(ve))
        m = in_val & panel_inc[list(BASELINES.values())].notna().all(axis=1)
        for _, (p, _, lvl) in data.items():
            m &= p["Total_Paid"].notna() & lvl.notna()
        masks[fold_name] = m
    return masks


def run_cross_validation(panel, X, level_log, common_by_fold, covid_mode,
                         panel_actual):
    """
    panel        : bu COVID modunun paneli (exclude'da COVID günleri NaN)
    panel_actual : COVID günleri maskelenmemiş panel (gerçek değerler)
    Baseline'lar yalnızca covid_mode == "include" iken hesaplanır.
    """
    results, store, best_iters = [], {}, {}
    labeled = panel["Total_Paid"].notna()

    for fold_name, (vs, ve) in FOLDS.items():
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        in_val = panel["transaction_date"].between(vs, ve)
        common = common_by_fold[fold_name]
        act = panel_actual.loc[common]
        print(f"\n{fold_name}: ortak değerlendirme satırı = {common.sum()}")

        if covid_mode == "include":
            for name, col in BASELINES.items():
                results.append({
                    "Fold": fold_name, "Model": name, "Kind": "baseline",
                    "mode": None, "feature_set": None, "covid_mode": None,
                    **evaluate(act["merchant_id"], act["Total_Paid"], act[col]),
                })
                store[(fold_name, name)] = {"pred": act[col].copy()}

        for mode in ("plain", "level"):
            tr = (panel["transaction_date"] < vs) & labeled
            va = in_val & labeled
            if mode == "level":
                tr &= level_log.notna()
                va &= level_log.notna()
            y_all = make_target(panel, level_log, mode)

            for fs in FEATURE_SETS:
                cols = columns_for(X, fs)
                name = model_name(mode, fs, covid_mode)

                model = make_model(n_estimators=2000)
                model.fit(
                    X.loc[tr, cols], y_all.loc[tr],
                    eval_set=[(X.loc[va, cols], y_all.loc[va])],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                curve = list(model.evals_result_["valid_0"]["l2"])
                best = int(model.best_iteration_ or len(curve))
                best_iters.setdefault(name, []).append(best)

                pred = pd.Series(
                    to_paid_scale(
                        model.predict(X.loc[va, cols]),
                        level_log.loc[va] if mode == "level" else 0.0, mode),
                    index=panel.index[va],
                )
                results.append({
                    "Fold": fold_name, "Model": name, "Kind": "lgbm",
                    "mode": mode, "feature_set": fs, "covid_mode": covid_mode,
                    **evaluate(act["merchant_id"], act["Total_Paid"],
                               pred.loc[common[common].index]),
                })
                store[(fold_name, name)] = {
                    "pred": pred, "best_iter": best, "curve": curve}

                flag = "  <-- UYARI: çok az ağaç" if best < 20 else ""
                print(f"  {name}: en iyi iterasyon = {best} | "
                      f"val l2: ilk {curve[0]:.4f} -> en iyi "
                      f"{curve[best - 1]:.4f}{flag}")

    return pd.DataFrame(results), store, best_iters


def ablation_report(results, factor, ref_value, title, filename):
    """
    Bir faktörün (özellik seti veya COVID modu) katkısını, diğer faktörler
    sabitken referans değere göre ölçer. Negatif fark = hatayı azaltıyor.
    """
    lg = results[results["Kind"] == "lgbm"]
    keys = ["mode", "feature_set", "covid_mode"]
    pivot = lg.pivot_table(index=keys, columns="Fold",
                           values="Macro SMAPE (%)")
    pivot["Ortalama"] = pivot.mean(axis=1)

    others = [k for k in keys if k != factor]
    ref = pivot["Ortalama"].xs(ref_value, level=factor)
    diffs = []
    for idx in pivot.index:
        key = tuple(idx[keys.index(o)] for o in others)
        diffs.append(pivot.loc[idx, "Ortalama"] - ref.loc[key])
    pivot[f"Fark (vs {ref_value})"] = diffs

    print(f"\n{title}")
    print("Macro SMAPE, düşük iyi. Negatif fark = faktör hatayı azaltıyor.")
    print(pivot.to_string(float_format=lambda x: f"{x:,.2f}"))
    pivot.to_csv(OUT_DIR / filename)


###################################################
# 5. GÖRSELLER
###################################################

def plot_monthly_volume(df, merchant_ids, colors):
    obs = df[df["transaction_date"] <= LAST_OBSERVED]
    monthly = (obs.groupby(["merchant_id",
                            pd.Grouper(key="transaction_date", freq="MS")])
                  ["Total_Paid"].sum().reset_index())

    fig, axes = merchant_grid(len(merchant_ids))
    for ax, mid in zip(axes, merchant_ids):
        m = monthly[monthly["merchant_id"] == mid]
        ax.axvspan(COVID_START, COVID_END,
                   color="#9aa0a6", alpha=0.15, lw=0)
        ax.fill_between(m["transaction_date"], m["Total_Paid"],
                        color=colors[mid], alpha=0.15, lw=0)
        ax.plot(m["transaction_date"], m["Total_Paid"],
                color=colors[mid], lw=1.8)
        ax.set_title(f"Merchant {mid}")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.yaxis.set_major_formatter(EngFormatter())

    spare_legend(axes, len(merchant_ids), [
        Patch(facecolor="#9aa0a6", alpha=0.3,
              label=f"COVID-19 dönemi ({COVID_START:%d.%m.%Y} - {COVID_END:%d.%m.%Y})"),
        Line2D([0], [0], color=INK, lw=1.8, label="Aylık toplam ödeme"),
    ], title="Açıklama")

    finish_figure(
        fig, "Aylık Ödeme Hacmi - İş Yeri Bazında", "01_monthly_volume.png",
        note="Kaynak: iyzico veri seti. Her panel kendi ölçeğindedir "
             "(iş yerleri arası doğrudan kıyas için uygun değildir).")


def plot_yearly_seasonality(df, merchant_ids):
    obs = df[df["transaction_date"] <= LAST_OBSERVED]
    fig, axes = merchant_grid(len(merchant_ids))
    ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    for ax, mid in zip(axes, merchant_ids):
        s = (obs[obs["merchant_id"] == mid]
             .set_index("transaction_date")["Total_Paid"])
        smooth = s.rolling(14, min_periods=7).mean()
        for year, color in YEAR_COLORS.items():
            part = smooth[smooth.index.year == year].dropna()
            if part.empty:
                continue
            ax.plot(part.index.dayofyear, part.values,
                    color=color, lw=1.6, label=str(year))
        ax.set_title(f"Merchant {mid}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(MONTH_TR, fontsize=8)
        ax.set_xlim(1, 366)
        ax.yaxis.set_major_formatter(EngFormatter())

    spare_legend(axes, len(merchant_ids), [
        Line2D([0], [0], color=c, lw=2, label=str(y))
        for y, c in YEAR_COLORS.items()
    ], title="Yıl")

    finish_figure(
        fig, "Yıllara Göre Mevsimsellik (14 Günlük Hareketli Ortalama)",
        "02_yearly_seasonality.png",
        note="Günlük Total_Paid, 14 günlük hareketli ortalama. 2020 verisi "
             f"{LAST_OBSERVED.date()} tarihine kadardır.")


def plot_weekday_heatmap(df, merchant_ids):
    obs = df[(df["transaction_date"] <= LAST_OBSERVED)
             & df["Total_Paid"].notna()].copy()
    obs["dow"] = obs["transaction_date"].dt.dayofweek
    mean_by_dow = obs.groupby(["merchant_id", "dow"])["Total_Paid"].mean()
    overall = obs.groupby("merchant_id")["Total_Paid"].mean()
    index = (mean_by_dow / overall).unstack().reindex(merchant_ids)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    vmax = max(1.01, np.nanmax(index.values))
    vmin = min(0.99, np.nanmin(index.values))
    im = ax.imshow(index.values, cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax))
    ax.grid(False)
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAY_TR)
    ax.set_yticks(range(len(merchant_ids)))
    ax.set_yticklabels([f"Merchant {m}" for m in merchant_ids])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            val = index.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Günlük ortalama / iş yeri ortalaması")
    cbar.outline.set_visible(False)

    finish_figure(
        fig, "Haftanın Günlerine Göre Ödeme Yoğunluğu",
        "03_weekday_heatmap.png",
        note="1.00 = iş yerinin genel günlük ortalaması. Kırmızı: ortalamanın "
             "üstü, mavi: altı.")


def plot_data_availability(panel, merchant_ids):
    obs = panel[panel["transaction_date"] <= LAST_OBSERVED].copy()
    obs["month"] = obs["transaction_date"].dt.to_period("M").dt.to_timestamp()
    share = (obs.groupby(["merchant_id", "month"])["is_observed"].mean()
                .unstack().reindex(merchant_ids))

    fig, ax = plt.subplots(figsize=(14, 4.2), layout="constrained")
    cmap = plt.get_cmap("Greens").copy()
    cmap.set_bad("#f1f3f4")
    im = ax.imshow(np.ma.masked_invalid(share.values), cmap=cmap,
                   vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    months = share.columns
    tick_pos = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([months[i].strftime("%b %y") for i in tick_pos])
    ax.set_yticks(range(len(merchant_ids)))
    ax.set_yticklabels([f"Merchant {m}" for m in merchant_ids])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Gözlenen gün oranı")
    cbar.outline.set_visible(False)

    finish_figure(
        fig, "Veri Kapsamı: Aylık Gözlenen Gün Oranı",
        "04_data_availability.png",
        note="Gri alan: iş yerinin ilk gözlem tarihinden önceki dönem "
             "(panele alınmaz).")


def plot_model_comparison(results):
    pivot = results.pivot_table(index="Model", columns="Fold",
                                values="Macro SMAPE (%)")
    lgbm_models = results.loc[results["Kind"] == "lgbm", "Model"].unique()
    top = (pivot.loc[pivot.index.isin(lgbm_models)].mean(axis=1)
           .nsmallest(12).index)
    keep = [m for m in pivot.index if m in BASELINES or m in set(top)]
    pivot = pivot.loc[keep]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    folds = list(pivot.columns)
    fold_colors = ["#A9C4D6", "#2A6F97", "#C8553D", "#8E5572"]

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(pivot) + 2),
                           layout="constrained")
    height = 0.8 / len(folds)
    y = np.arange(len(pivot))
    for k, fold in enumerate(folds):
        pos = y + (k - (len(folds) - 1) / 2) * height
        bars = ax.barh(pos, pivot[fold].values, height=height * 0.92,
                       color=fold_colors[k % len(fold_colors)], label=fold)
        for bar, val in zip(bars, pivot[fold].values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Macro SMAPE (%) - düşük daha iyi")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    finish_figure(
        fig, "Model Karşılaştırması: Baseline'lar ve LightGBM Varyantları",
        "05_model_comparison.png",
        note="Baseline'lar ve en iyi 12 LightGBM varyantı. Tüm modeller her "
             "fold'da aynı satırlar üzerinde "
             "değerlendirilmiştir. Macro SMAPE: iş yeri bazlı SMAPE'lerin "
             "ortalaması.")


def plot_validation_forecast(panel, store, fold_name, name, merchant_ids,
                             colors):
    pred = store[(fold_name, name)]["pred"]
    frame = panel.loc[pred.index, ["merchant_id", "transaction_date",
                                   "Total_Paid"]].copy()
    frame["pred"] = pred.values
    frame = frame.dropna(subset=["Total_Paid", "pred"])

    fig, axes = merchant_grid(len(merchant_ids))
    for ax, mid in zip(axes, merchant_ids):
        m = frame[frame["merchant_id"] == mid]
        if m.empty:
            ax.axis("off")
            continue
        ax.plot(m["transaction_date"], m["Total_Paid"],
                color=INK, lw=1.2, alpha=0.85)
        ax.plot(m["transaction_date"], m["pred"],
                color=colors[mid], lw=1.6)
        score = smape(m["Total_Paid"], m["pred"])
        ax.set_title(f"Merchant {mid}")
        ax.text(0.99, 0.95, f"SMAPE {score:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=MUTED)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.yaxis.set_major_formatter(EngFormatter())

    spare_legend(axes, len(merchant_ids), [
        Line2D([0], [0], color=INK, lw=1.4, label="Gerçek"),
        Line2D([0], [0], color=MERCHANT_COLORS[0], lw=1.8, label="Tahmin"),
    ], title=name)

    safe = fold_name.split(" (")[0].replace(" ", "_").lower()
    finish_figure(
        fig, f"Validation: Gerçek vs Tahmin - {fold_name}",
        f"06_validation_{safe}.png",
        note="Günlük Total_Paid. Tahminler 92 günlük ufukla, yalnızca "
             "tahmin başlangıcında bilinen bilgilerle üretilmiştir.")


def plot_learning_curves(store, covid_mode):
    """Early stopping neden erken durdu? Validation l2 eğrilerine bak."""
    folds = list(FOLDS)
    modes = ("plain", "level")
    palette = ["#9aa0a6", "#2A6F97", "#C8553D"]
    fs_colors = dict(zip(FEATURE_SETS, palette))

    fig, axes = plt.subplots(len(modes), len(folds), figsize=(13, 7),
                             layout="constrained", squeeze=False)
    for i, mode in enumerate(modes):
        for j, fold in enumerate(folds):
            ax = axes[i, j]
            for fs in FEATURE_SETS:
                entry = store[(fold, model_name(mode, fs, covid_mode))]
                curve = np.asarray(entry["curve"])
                x = np.arange(1, len(curve) + 1)
                ax.plot(x, curve, color=fs_colors[fs], lw=1.5, label=fs)
                b = entry["best_iter"]
                ax.scatter([b], [curve[b - 1]], color=fs_colors[fs],
                           s=28, zorder=3)
            ax.set_xscale("log")
            ax.set_title(f"{MODEL_LABELS[mode]} - {fold}", fontsize=10)
            ax.set_xlabel("Ağaç sayısı (log)")
            ax.set_ylabel("Validation l2")
            if i == 0 and j == 0:
                ax.legend(title="Özellik seti")

    finish_figure(
        fig, f"Öğrenme Eğrileri - COVID modu: {covid_mode}",
        f"09_learning_curves_{covid_mode}.png",
        note="Nokta: early stopping'in seçtiği iterasyon. Eğri en başta en "
             "düşükse model sabit tahminden daha fazlasını öğrenememiştir.")


def plot_feature_importance(model, columns, top_n=15):
    gain = pd.Series(model.booster_.feature_importance("gain"),
                     index=columns).sort_values(ascending=False)
    gain = 100 * gain / gain.sum()
    top = gain.head(top_n).iloc[::-1]
    calendar_like = set(CALENDAR_COLS)
    bar_colors = ["#C8553D" if c in calendar_like else "#2A6F97"
                  for c in top.index]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(top.index, top.values, color=bar_colors)
    for i, v in enumerate(top.values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Toplam gain içindeki pay (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Patch(color="#C8553D", label="Takvim / tatil özelliği"),
        Patch(color="#2A6F97", label="Geçmiş değer özelliği"),
    ], loc="lower right")

    finish_figure(fig, f"En Önemli {top_n} Özellik (Final Model)",
                  "07_feature_importance.png")


def plot_test_forecast(panel, forecast, merchant_ids, colors,
                       history_days=180):
    hist_start = LAST_OBSERVED - pd.Timedelta(days=history_days)
    fig, axes = merchant_grid(len(merchant_ids))

    for ax, mid in zip(axes, merchant_ids):
        h = panel[(panel["merchant_id"] == mid)
                  & (panel["transaction_date"] > hist_start)
                  & (panel["transaction_date"] <= LAST_OBSERVED)]
        f = forecast[forecast["merchant_id"] == mid]
        ax.axvspan(LAST_OBSERVED, FORECAST_END, color=colors[mid],
                   alpha=0.07, lw=0)
        ax.plot(h["transaction_date"], h["Total_Paid"],
                color=INK, lw=1.2, alpha=0.85)
        ax.plot(f["transaction_date"], f["predicted_total_paid"],
                color=colors[mid], lw=1.8)
        for bf in BLACK_FRIDAYS[-1:]:
            ax.axvline(pd.Timestamp(bf), color=MUTED, lw=0.9, ls=":")
        ax.axvline(LAST_OBSERVED, color=MUTED, lw=0.9, ls="--")
        ax.set_title(f"Merchant {mid}")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.yaxis.set_major_formatter(EngFormatter())

    spare_legend(axes, len(merchant_ids), [
        Line2D([0], [0], color=INK, lw=1.4, label="Gerçekleşen (son 180 gün)"),
        Line2D([0], [0], color=MERCHANT_COLORS[0], lw=1.8,
               label="Tahmin (92 gün)"),
        Line2D([0], [0], color=MUTED, lw=1, ls="--", label="Tahmin başlangıcı"),
        Line2D([0], [0], color=MUTED, lw=1, ls=":", label="Black Friday 2020"),
    ], title="Açıklama")

    finish_figure(
        fig, "Ekim-Aralık 2020 Ödeme Hacmi Tahmini",
        "08_test_forecast.png",
        note="Tahmin, 30 Eylül 2020'ye kadar bilinen verilerle üretilmiştir.")


###################################################
# 6. ANA AKIŞ
###################################################

def main():
    global FIG_DIR, OUT_DIR
    set_style()
    if "include" not in COVID_MODES:
        raise ValueError('COVID_MODES içinde "include" bulunmalı '
                         "(baseline ve gerçek değerler için referans).")

    data_path = find_data_file()
    base_dir = (data_path.parent.parent if data_path.parent.name == "data"
                else data_path.parent)
    FIG_DIR = base_dir / "figures"
    OUT_DIR = base_dir / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    print("Çıktılar şuraya yazılacak:", base_dir)

    # --- Veri ---
    df = load_data(data_path)
    df, test_truth = split_off_future(df)
    merchant_ids = sorted(df["merchant_id"].unique())
    colors = {m: MERCHANT_COLORS[i % len(MERCHANT_COLORS)]
              for i, m in enumerate(merchant_ids)}

    summary = df.groupby("merchant_id").agg(
        observed_days=("Total_Paid", "count"),
        total_transactions=("Total_Transaction", "sum"),
        total_paid=("Total_Paid", "sum"),
        average_daily_paid=("Total_Paid", "mean"),
    ).sort_values("total_paid", ascending=False)
    print("\nMERCHANT SUMMARY")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    # --- Panel ve özellikler (her COVID modu için ayrı) ---
    panel_base = build_panel(df)
    data_quality_report(panel_base)
    panel_base = add_covid_flag(add_calendar_features(panel_base))

    data = {}
    for cmode in COVID_MODES:
        p = panel_base.copy()
        if cmode == "exclude":
            p.loc[p["is_covid"] == 1, "Total_Paid"] = np.nan
        p = add_history_features(p)
        data[cmode] = (p, build_design_matrix(p, cmode),
                       np.log1p(p["paid_roll_mean_91"]))
    panel_actual = data["include"][0]   # COVID günleri maskelenmemiş gerçek veri
    n_covid = int(panel_base["is_covid"].sum())
    print(f"\nPANEL: {panel_actual.shape} | COVID günü (iş yeri x gün): "
          f"{n_covid} | COVID modları: {COVID_MODES}")

    # --- EDA görselleri ---
    plot_monthly_volume(df, merchant_ids, colors)
    plot_yearly_seasonality(df, merchant_ids)
    plot_weekday_heatmap(df, merchant_ids)
    plot_data_availability(panel_base, merchant_ids)

    # --- Çapraz doğrulama (tüm COVID modları, aynı satırlarda) ---
    candidates = build_candidates()
    common_by_fold = fold_common_masks(data)
    all_results, store, best_iters = [], {}, {}
    for cmode in COVID_MODES:
        p, X, lvl = data[cmode]
        print(f"\n========== COVID modu: {cmode} ==========")
        res, st, bi = run_cross_validation(p, X, lvl, common_by_fold,
                                           cmode, panel_actual)
        all_results.append(res)
        store.update(st)
        best_iters.update(bi)
    results = pd.concat(all_results, ignore_index=True)

    table = results.drop(columns=["Kind", "mode", "feature_set",
                                  "covid_mode"]).sort_values(
        ["Fold", "Macro SMAPE (%)"])
    print("\nVALIDATION SONUÇLARI (ortak satırlar)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    table.to_csv(OUT_DIR / "validation_results.csv", index=False)

    ablation_report(results, "feature_set", list(FEATURE_SETS)[0],
                    "TATİL / ALIŞVERİŞ ÖZELLİĞİ ABLATION",
                    "holiday_ablation.csv")
    ablation_report(results, "covid_mode", "include",
                    "COVID MODU ABLATION", "covid_ablation.csv")
    plot_model_comparison(results)
    for cmode in COVID_MODES:
        plot_learning_curves(store, cmode)

    # --- Final seçim: baseline'lar dahil TÜM adaylar, aynı metrik ---
    mean_scores = (results.groupby("Model")["Macro SMAPE (%)"].mean()
                   .sort_values())
    print("\nEN İYİ 15 ADAY - iki fold ortalaması (Macro SMAPE, düşük iyi)")
    print(mean_scores.head(15).to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\nBASELINE'LAR")
    print(mean_scores[mean_scores.index.isin(BASELINES)]
          .to_string(float_format=lambda x: f"{x:,.2f}"))

    by_name = {c["name"]: c for c in candidates}

    def complexity(name):
        c = by_name[name]
        if c["kind"] == "baseline":
            return 0
        return (1 + 2 * list(FEATURE_SETS).index(c["fs"])
                + (0 if c["covid"] == "include" else 1))

    near = mean_scores[mean_scores <= mean_scores.iloc[0] + SELECTION_TOLERANCE]
    chosen_name = min(near.index, key=lambda n: (complexity(n), mean_scores[n]))
    chosen = by_name[chosen_name]
    print(f"\nSeçilen model: {chosen_name} "
          f"(en iyi skorun {SELECTION_TOLERANCE} puan içindeki en basit aday)")

    for fold_name in FOLDS:
        plot_validation_forecast(panel_actual, store, fold_name, chosen_name,
                                 merchant_ids, colors)

    # --- Final tahmin ---
    final_model, final_cols = None, None

    if chosen["kind"] == "baseline":
        panel, X, level_log = data["include"]
        test_mask = ((panel["transaction_date"] > LAST_OBSERVED)
                     & (panel["transaction_date"] <= FORECAST_END))
        forecast = panel.loc[test_mask, ["merchant_id",
                                         "transaction_date"]].copy()
        forecast["predicted_total_paid"] = (
            panel.loc[test_mask, chosen["col"]].to_numpy())
        n_bad = int(forecast["predicted_total_paid"].isna().sum())
        if n_bad:
            print(f"UYARI: {n_bad} test satırında baseline değeri eksik; "
                  "bu satırlar tahmin dışı kalacak.")
        forecast = forecast.dropna(subset=["predicted_total_paid"])
    else:
        mode, fs = chosen["mode"], chosen["fs"]
        panel, X, level_log = data[chosen["covid"]]
        final_cols = columns_for(X, fs)
        test_mask = ((panel["transaction_date"] > LAST_OBSERVED)
                     & (panel["transaction_date"] <= FORECAST_END))

        labeled = (panel["Total_Paid"].notna()
                   & (panel["transaction_date"] <= LAST_OBSERVED))
        if mode == "level":
            labeled &= level_log.notna()

        # Fold'larda bulunan en iyi ağaç sayılarının medyanı
        n_estimators = max(1, int(round(np.median(best_iters[chosen_name]))))
        print("Fold'lardaki en iyi iterasyonlar:", best_iters[chosen_name],
              "-> final ağaç sayısı:", n_estimators)
        if n_estimators < 20:
            print("UYARI: Final model çok az ağaçla eğitiliyor. Öğrenme "
                  "eğrilerine (09_learning_curves_*.png) bakıp baseline'a "
                  "göre kazancın gerçek olup olmadığını kontrol et.")

        y_all = make_target(panel, level_log, mode)
        final_model = make_model(n_estimators)
        final_model.fit(X.loc[labeled, final_cols], y_all.loc[labeled])

        if mode == "level":
            n_bad = int(level_log.loc[test_mask].isna().sum())
            if n_bad:
                print(f"UYARI: {n_bad} test satırında seviye özelliği "
                      "eksik; bu satırlar tahmin dışı kalacak.")
            test_mask &= level_log.notna()

        forecast = panel.loc[test_mask, ["merchant_id",
                                         "transaction_date"]].copy()
        forecast["predicted_total_paid"] = to_paid_scale(
            final_model.predict(X.loc[test_mask, final_cols]),
            level_log.loc[test_mask] if mode == "level" else 0.0, mode)

    forecast.to_csv(OUT_DIR / "iyzico_test_forecast.csv", index=False)
    print("\nTEST TAHMİNİ:", forecast.shape,
          "-> outputs/iyzico_test_forecast.csv")
    print(forecast.groupby("merchant_id")["predicted_total_paid"]
          .agg(["count", "sum", "mean"])
          .to_string(float_format=lambda x: f"{x:,.2f}"))

    if final_model is not None:
        plot_feature_importance(final_model, final_cols)
    plot_test_forecast(panel_actual, forecast, merchant_ids, colors)

    # --- Test etiketleri varsa skor ---
    if test_truth is not None:
        joined = forecast.set_index(["merchant_id", "transaction_date"]).join(
            test_truth.rename("actual"), how="inner")
        if len(joined):
            scores = evaluate(
                joined.index.get_level_values("merchant_id"),
                joined["actual"], joined["predicted_total_paid"])
            print("\nTEST SKORU:", {k: round(v, 2) for k, v in scores.items()})


if __name__ == "__main__":
    main()
