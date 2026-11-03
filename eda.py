"""
EDA очищенного датасета + анализ позиции «Меркур Авто».
Ключевые цифры → консоль и eda_results.txt; графики → папка eda_plots/.

Запуск:  python eda.py
Вход:    autokz2019_cleaned.csv  (результат clean.py)
Выход:   eda_results.txt, eda_plots/*.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_DIR = "eda_plots"
CURRENCY = "KZT"


# ---------- Загрузка ----------
def load_clean(path: str = "autokz2019_cleaned.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def check_price_consistency(df: pd.DataFrame) -> None:
    """Проверка: Price — за единицу или за строку. Только по рознице (Quantity=1)."""
    sub = df[df["Quantity"] == 1]
    diff = (sub["Price"] * sub["Quantity"] - sub["SaleAmount"]).abs()
    denom = sub["SaleAmount"].abs().replace(0, np.nan)
    rel = (diff / denom).dropna()
    print("\n[check] Price * Quantity vs SaleAmount (только Quantity=1):")
    print(f"  строк проверено: {len(sub)}")
    print(f"  max |diff|              = {diff.max():,.2f}")
    print(f"  доля строк с diff > 1% = {(rel > 0.01).mean() * 100:.2f}%")
    if (rel > 0.01).mean() > 0.05:
        print("  ⚠ Price, похоже, НЕ за единицу — средний чек считаем "
              "как SaleAmount / Quantity")
    else:
        print("  ✓ Price — за единицу")


# ---------- Рынок ----------
def market_overview(df: pd.DataFrame) -> dict:
    res = {}

    brand = df.groupby("Brand", observed=True).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )
    brand["avg_price_unit"] = brand["revenue"] / brand["units"].replace(0, np.nan)
    brand["share_units_pct"] = brand["units"] / brand["units"].sum() * 100
    brand["share_revenue_pct"] = brand["revenue"] / brand["revenue"].sum() * 100
    res["brand_sales"] = brand.sort_values("units", ascending=False)

    df_valid = df[df["SaleDate"].notna()].copy()
    df_valid["Month"] = df_valid["SaleDate"].dt.to_period("M").astype(str)
    res["monthly_sales"] = df_valid.groupby("Month").agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )

    res["country_sales"] = df.groupby("Country", observed=True).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    ).sort_values("units", ascending=False)

    res["fuel_sales"] = df.groupby("FuelType", observed=True, dropna=False).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )
    res["drive_sales"] = df.groupby("DriveType", observed=True, dropna=False).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )
    res["trans_sales"] = df.groupby("Transmission", observed=True, dropna=False).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )
    res["region_sales"] = df.groupby("Region", observed=True, dropna=False).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    ).sort_values("units", ascending=False)
    res["class_sales"] = df.groupby("Class2013", observed=True, dropna=False).agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    ).sort_values("units", ascending=False)

    total_qty = df["Quantity"].sum()
    total_rev = df["SaleAmount"].sum()
    res["market_avg_price_weighted"] = total_rev / total_qty if total_qty else np.nan
    res["market_avg_price_simple"] = df["Price"].mean()
    res["market_median_price"] = df["Price"].median()

    res["price_desc"] = df["Price"].describe()
    res["engine_desc"] = df["EngineVolume"].describe()
    return res


# ---------- Меркур ----------
def mercur_analysis(df: pd.DataFrame, market: dict) -> dict:
    mercur = df[df["Company"] == "Mercur Auto"].copy()
    res = {}

    # Диагностика NaT
    nat = mercur["SaleDate"].isna().sum()
    if nat:
        print(f"\n⚠ Mercur: {nat} строк с NaT в SaleDate — "
              f"выпадут из monthly. Проверьте clean.py.")
    res["n_nat"] = int(nat)

    res["records"] = len(mercur)
    res["units"] = mercur["Quantity"].sum()
    res["revenue"] = mercur["SaleAmount"].sum()

    total_units = df["Quantity"].sum()
    total_revenue = df["SaleAmount"].sum()
    res["share_units"] = res["units"] / total_units * 100 if total_units else np.nan
    res["share_revenue"] = res["revenue"] / total_revenue * 100 if total_revenue else np.nan

    res["avg_price_weighted"] = (
        res["revenue"] / res["units"] if res["units"] else np.nan
    )
    res["avg_price_diff_pct"] = (
        (res["avg_price_weighted"] - market["market_avg_price_weighted"])
        / market["market_avg_price_weighted"] * 100
    )

    def _agg(group_col: str) -> pd.DataFrame:
        g = mercur.groupby(group_col, observed=True, dropna=False).agg(
            units=("Quantity", "sum"),
            revenue=("SaleAmount", "sum"),
        )
        g["avg_price_unit"] = g["revenue"] / g["units"].replace(0, np.nan)
        g = g.sort_values("revenue", ascending=False)
        return g

    res["by_brand"] = _agg("Brand")
    res["by_region"] = _agg("Region")
    res["by_class"] = _agg("Class2013")
    res["by_fuel"] = _agg("FuelType")
    res["by_drive"] = _agg("DriveType")

    # Доля Меркура внутри брендов (топ-15 рынка)
    rows = []
    for brand in market["brand_sales"].head(15).index:
        m_units = market["brand_sales"].loc[brand, "units"]
        mer_units = res["by_brand"]["units"].get(brand, 0)
        rows.append({
            "Brand": brand,
            "market_units": m_units,
            "mercur_units": mer_units,
            "mercur_share_in_brand_pct": (
                mer_units / m_units * 100 if m_units else 0
            ),
        })
    res["brand_comparison"] = pd.DataFrame(rows)

    # Доля Меркура внутри регионов — по объёму И по выручке
    region_qty = df.groupby("Region", observed=True)["Quantity"].sum()
    region_rev = df.groupby("Region", observed=True)["SaleAmount"].sum()
    region_m_qty = mercur.groupby("Region", observed=True)["Quantity"].sum()
    region_m_rev = mercur.groupby("Region", observed=True)["SaleAmount"].sum()
    res["region_share_units"] = (
        (region_m_qty / region_qty * 100).dropna().sort_values(ascending=False)
    )
    res["region_share_revenue"] = (
        (region_m_rev / region_rev * 100).dropna().sort_values(ascending=False)
    )

    # Доля Меркура внутри классов — по объёму И по выручке
    class_qty = df.groupby("Class2013", observed=True, dropna=False)["Quantity"].sum()
    class_rev = df.groupby("Class2013", observed=True, dropna=False)["SaleAmount"].sum()
    class_m_qty = mercur.groupby("Class2013", observed=True, dropna=False)["Quantity"].sum()
    class_m_rev = mercur.groupby("Class2013", observed=True, dropna=False)["SaleAmount"].sum()
    res["class_share_units"] = (
        (class_m_qty / class_qty * 100).dropna().sort_values(ascending=False)
    )
    res["class_share_revenue"] = (
        (class_m_rev / class_rev * 100).dropna().sort_values(ascending=False)
    )

    # Месячная динамика Меркура
    mv = mercur[mercur["SaleDate"].notna()].copy()
    mv["Month"] = mv["SaleDate"].dt.to_period("M").astype(str)
    res["monthly"] = mv.groupby("Month").agg(
        units=("Quantity", "sum"),
        revenue=("SaleAmount", "sum"),
    )
    return res


# ---------- Визуализации ----------
def _save(fig, name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_plots(df: pd.DataFrame, market: dict, mercur: dict) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Топ-15 брендов
    top = market["brand_sales"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index.astype(str), top["units"])
    ax.set_xlabel("Продано, шт")
    ax.set_title("Топ-15 брендов по объёму продаж")
    _save(fig, "01_top_brands.png")

    # 2. Динамика по месяцам
    ms = market["monthly_sales"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ms.index.astype(str), ms["units"], marker="o")
    ax.set_ylabel("Продано, шт")
    ax.set_title("Динамика продаж по месяцам")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "02_monthly.png")

    # 3. Распределение цены (до 99-го перцентиля)
    p99 = df["Price"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.loc[df["Price"] <= p99, "Price"].dropna(), bins=50)
    ax.set_xlabel(f"Цена, {CURRENCY}")
    ax.set_title("Распределение цены (до 99-го перцентиля)")
    _save(fig, "03_price_hist.png")

    # 4. Объём двигателя — шаг 0.1 л, чтобы не было «дыр»
    ev = df["EngineVolume"].dropna()
    if len(ev):
        ev = ev[(ev > 0) & (ev <= ev.quantile(0.99))]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ev, bins=np.arange(1.0, 5.5, 0.1))
        ax.set_xlabel("Объём двигателя, л")
        ax.set_title("Распределение объёма двигателя")
        _save(fig, "04_engine_hist.png")

    # 5. Топливо / привод / КПП — с подписями значений
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes,
        ["fuel_sales", "drive_sales", "trans_sales"],
        ["Топливо", "Привод", "КПП"],
    ):
        s = market[col]["units"]
        s.plot.bar(ax=ax, title=title)
        ax.set_ylabel("Продано, шт")
        for i, v in enumerate(s):
            if pd.notna(v):
                ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    _save(fig, "05_categories.png")

    # 6. Топ-10 регионов
    top_r = market["region_sales"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_r.index.astype(str), top_r["units"])
    ax.set_xlabel("Продано, шт")
    ax.set_title("Топ-10 регионов")
    _save(fig, "06_top_regions.png")

    # 7. Меркур vs рынок — по брендам, где Меркур присутствует
    mer_brands = mercur["by_brand"].index[:10]
    if len(mer_brands):
        bc = market["brand_sales"].loc[mer_brands].copy()
        bc["mercur_units"] = (
            mercur["by_brand"]["units"].reindex(mer_brands).fillna(0)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bc))
        w = 0.4
        ax.bar(x - w / 2, bc["units"], w, label="Рынок")
        ax.bar(x + w / 2, bc["mercur_units"], w, label="Mercur Auto")
        ax.set_xticks(x)
        ax.set_xticklabels(bc.index.astype(str), rotation=45, ha="right")
        ax.set_ylabel("Продано, шт")
        ax.set_title("Mercur Auto vs рынок: бренды Меркура")
        ax.legend()
        _save(fig, "07_mercur_vs_market.png")

    print(f"Графики сохранены в {PLOT_DIR}/")


# ---------- Текстовый отчёт ----------
def save_report(market: dict, mercur: dict, path: str = "eda_results.txt") -> None:
    lines = []
    lines.append("EDA RESULTS\n" + "=" * 60 + "\n")

    lines.append("1. РЫНОК ПО БРЕНДАМ (топ-15)\n")
    lines.append(market["brand_sales"].head(15).to_string() + "\n\n")

    lines.append("2. РЫНОК ПО МЕСЯЦАМ\n")
    lines.append(market["monthly_sales"].to_string() + "\n\n")

    lines.append("3. РЫНОК ПО СТРАНАМ\n")
    lines.append(market["country_sales"].to_string() + "\n\n")

    lines.append("4. ТОПЛИВО / ПРИВОД / КПП\n")
    lines.append(market["fuel_sales"].to_string() + "\n")
    lines.append(market["drive_sales"].to_string() + "\n")
    lines.append(market["trans_sales"].to_string() + "\n\n")

    lines.append("5. ТОП-10 РЕГИОНОВ\n")
    lines.append(market["region_sales"].head(10).to_string() + "\n\n")

    lines.append("6. КЛАССЫ 2013\n")
    lines.append(market["class_sales"].to_string() + "\n\n")

    lines.append("7. ЦЕНЫ (describe)\n")
    lines.append(market["price_desc"].to_string() + "\n")
    lines.append(
        f"Взвешенный средний чек рынка: "
        f"{market['market_avg_price_weighted']:,.0f} {CURRENCY}\n"
    )
    lines.append(
        f"Простой средний чек рынка: "
        f"{market['market_avg_price_simple']:,.0f} {CURRENCY}\n"
    )
    lines.append(f"Медиана: {market['market_median_price']:,.0f} {CURRENCY}\n\n")

    lines.append("8. ОБЪЁМ ДВИГАТЕЛЯ (describe)\n")
    lines.append(market["engine_desc"].to_string() + "\n\n")

    lines.append("=" * 60 + "\nMERCUR AUTO\n" + "=" * 60 + "\n")
    lines.append(f"NaT в SaleDate: {mercur['n_nat']}\n")
    lines.append(f"Записей: {mercur['records']}\n")
    lines.append(f"Единиц: {mercur['units']:.0f}\n")
    lines.append(f"Выручка: {mercur['revenue']:,.0f} {CURRENCY}\n")
    lines.append(f"Доля по объёму: {mercur['share_units']:.2f}%\n")
    lines.append(f"Доля по выручке: {mercur['share_revenue']:.2f}%\n")
    lines.append(
        f"Средняя цена единицы: "
        f"{mercur['avg_price_weighted']:,.0f} {CURRENCY}\n"
    )
    lines.append(f"Отличие от рынка: {mercur['avg_price_diff_pct']:+.1f}%\n\n")

    lines.append("По брендам:\n" + mercur["by_brand"].to_string() + "\n\n")
    lines.append("По регионам:\n" + mercur["by_region"].to_string() + "\n\n")
    lines.append(
        "Доля Mercur внутри регионов по объёму, %:\n"
        + mercur["region_share_units"].to_string() + "\n\n"
    )
    lines.append(
        "Доля Mercur внутри регионов по выручке, %:\n"
        + mercur["region_share_revenue"].to_string() + "\n\n"
    )
    lines.append("По классам:\n" + mercur["by_class"].to_string() + "\n\n")
    lines.append(
        "Доля Mercur внутри классов по объёму, %:\n"
        + mercur["class_share_units"].to_string() + "\n\n"
    )
    lines.append(
        "Доля Mercur внутри классов по выручке, %:\n"
        + mercur["class_share_revenue"].to_string() + "\n\n"
    )
    lines.append("По топливу:\n" + mercur["by_fuel"].to_string() + "\n\n")
    lines.append("По приводу:\n" + mercur["by_drive"].to_string() + "\n\n")
    lines.append("Сравнение по брендам (Mercur vs рынок):\n")
    lines.append(mercur["brand_comparison"].to_string(index=False) + "\n\n")
    lines.append("Месячная динамика Mercur:\n" + mercur["monthly"].to_string() + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Отчёт сохранён: {path}")


# ---------- main ----------
def main():
    df = load_clean()
    check_price_consistency(df)

    market = market_overview(df)
    mercur = mercur_analysis(df, market)

    print("\n=== РЫНОК: ТОП-10 БРЕНДОВ ===")
    print(market["brand_sales"].head(10).to_string())

    print("\n=== MERCUR AUTO ===")
    print(f"NaT в SaleDate: {mercur['n_nat']}")
    print(f"Единиц: {mercur['units']:.0f}, "
          f"выручка: {mercur['revenue']:,.0f} {CURRENCY}")
    print(f"Доля: {mercur['share_units']:.2f}% (объём), "
          f"{mercur['share_revenue']:.2f}% (выручка)")
    print(f"Средняя цена единицы Mercur: "
          f"{mercur['avg_price_weighted']:,.0f} {CURRENCY} "
          f"(рынок: {market['market_avg_price_weighted']:,.0f}, "
          f"отличие {mercur['avg_price_diff_pct']:+.1f}%)")
    print("\nMercur по брендам:\n", mercur["by_brand"].to_string())
    print("\nMercur: доля внутри регионов по объёму, %:\n",
          mercur["region_share_units"].to_string())
    print("\nMercur: доля внутри классов по объёму, %:\n",
          mercur["class_share_units"].to_string())

    save_report(market, mercur)
    make_plots(df, market, mercur)


if __name__ == "__main__":
    main()