"""
Очистка данных о продажах автомобилей в Казахстане за 2019 год.
Курсовой проект «Предподготовка и анализ данных».

Порядок обработки:
1) чтение CSV;
2) удаление ненужных колонок (строго по ТЗ);
3) нормализация категорий (Страна, Топливо, Привод, Трансмиссия,
   Объём двигателя, Компания, Регион);
4) SaleDate — ДО агрегации (последний день месяца);
5) флаги аномалий (возвраты / опт) — ДО агрегации;
6) агрегация дублей с суммированием Quantity/SaleAmount
   (drop_duplicates НЕ используется — он теряет продажи);
7) финальные типы (datetime / category / Int64);
8) контроль: суммы до/после, пропуски, dtypes, потери по брендам.
"""
import pandas as pd
import numpy as np
import re


RAW_COLS = [
    "Year", "Month", "Company", "Brand", "Model", "Modification",
    "ProductionYear", "Country", "FuelType", "EngineVolume",
    "Transmission", "DriveType", "Segment", "Region", "DealerName",
    "ClientType", "PaymentType", "Quantity", "Price", "SaleAmount",
    "Region2", "Segment2013", "Class2013", "SegmentEng", "Localization",
]

COLS_TO_DROP = [
    "DealerName", "PaymentType", "Segment", "SegmentEng",
    "Localization", "ClientType", "Modification",
]

# Группировка при агрегации дублей.
# Используем SaleDate (создаётся ДО агрегации), а не Year/Month.
GROUP_COLS = [
    "SaleDate", "Company", "Brand", "Model",
    "ProductionYear", "Country", "FuelType", "EngineVolume",
    "Transmission", "DriveType", "Region", "Region2",
    "Segment2013", "Class2013",
]


# ---------- 1. Загрузка ----------
def load_raw(path: str = "autokz2019.csv") -> pd.DataFrame:
    """Чтение CSV с учётом разделителя ';' и десятичной запятой."""
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    if len(df.columns) != len(RAW_COLS):
        raise ValueError(
            f"Ожидалось {len(RAW_COLS)} колонок, получено {len(df.columns)}"
        )
    df.columns = RAW_COLS
    print(f"Исходный размер: {df.shape}")
    print("Исходные колонки:", list(df.columns))
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление полей, которые по ТЗ не несут аналитической ценности."""
    present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=present)
    print(f"Удалены столбцы: {present}")
    print(f"После удаления столбцов: {df.shape}")
    return df


# ---------- 2. Нормализация категорий ----------
def encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """Страна производства → ALPHA-3."""
    country_map = {
        "Германия": "DEU", "США": "USA", "Казахстан": "KAZ",
        "Республика Казахстан": "KAZ", "Япония": "JPN",
        "Южная Корея": "KOR", "Корея": "KOR", "Франция": "FRA",
        "Италия": "ITA", "Швеция": "SWE", "Китай": "CHN",
        "Великобритания": "GBR", "UK": "GBR", "Австрия": "AUT",
        "Российская Федерация": "RUS", "Россия": "RUS",
        "Таиланд": "THA", "Узбекистан": "UZB", "Венгрия": "HUN",
        "Турция": "TUR", "Испания": "ESP", "Нидерланды": "NLD",
        "Польша": "POL", "Белоруссия": "BLR", "Бельгия": "BEL",
    }

    def _norm(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip()
        if not s:
            return "Unknown"
        if s in country_map:
            return country_map[s]
        print(f"  [Country] нераспознано: {s!r}")
        return "Unknown"

    df["Country"] = df["Country"].apply(_norm)
    print("Country:\n", df["Country"].value_counts().to_string())
    return df


def encode_fuel(df: pd.DataFrame) -> pd.DataFrame:
    """Бензин→F, Дизель→D, Электро→E, Гибрид→HYB, остальное → Unknown."""
    fuel_map = {
        "бензин": "F", "дизель": "D",
        "электро": "E", "электричество": "E",
        "гибрид": "HYB",
    }

    def _norm(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        return fuel_map.get(s, "Unknown")

    df["FuelType"] = df["FuelType"].apply(_norm)
    print("FuelType:\n", df["FuelType"].value_counts().to_string())
    return df


def normalize_drive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Привод → FWD / RWD / 4WD / Unknown.
    '4x2' интерпретируется как RWD: в выборке это преимущественно
    заднеприводные модели; иначе доля Unknown достигает 42%.
    """
    full = {
        "полный", "4wd", "awd", "quattro", "4motion", "полный привод",
        "4x4", "4 x 4", "4wd permanent", "4х4", "full-time 4wd",
    }
    front = {
        "передний", "fwd", "front-wheel drive", "ff", "передний привод",
        "2wd", "2 wd",
    }
    rear = {
        "задний", "rwd", "rear-wheel drive", "задний привод",
    }

    def _norm(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if s in {"", "0", "#н/д"}:
            return "Unknown"
        if s in {"4х2", "4x2", "4х2.2", "4x2.2"}:
            return "RWD"
        if s in full:
            return "4WD"
        if s in front:
            return "FWD"
        if s in rear:
            return "RWD"
        if any(p in s for p in ["4х", "4x"]) and "4х2" not in s and "4x2" not in s:
            return "4WD"
        return "Unknown"

    df["DriveType"] = df["DriveType"].apply(_norm)
    unk = (df["DriveType"] == "Unknown").sum()
    print(f"DriveType Unknown: {unk} ({unk / len(df) * 100:.1f}%)")
    print("DriveType:\n", df["DriveType"].value_counts().to_string())
    return df


def normalize_transmission(df: pd.DataFrame) -> pd.DataFrame:
    """Трансмиссия → Автомат / Механика / Unknown."""
    manual_patterns = ["мт", "мкпп", "мкп", "мех", "механика", "mt", "m/t"]
    auto_patterns = [
        "ат", "акпп", "акп", "автомат", "at", "dct", "dsg",
        "pdk", "powershift", "tronic", "steptronic", "tiptronic",
        "cvt", "вари", "a/t", "авт",
    ]

    def _norm(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if s in {"", "0", "#н/д"}:
            return "Unknown"
        if s in {"4wd", "awd", "4x4", "4х4"}:
            return "Unknown"
        if any(p in s for p in manual_patterns):
            return "Механика"
        if any(p in s for p in auto_patterns):
            return "Автомат"
        return "Unknown"

    df["Transmission"] = df["Transmission"].apply(_norm)
    print("Transmission:\n", df["Transmission"].value_counts().to_string())
    return df


# ---------- 3. Объём двигателя ----------
def clean_engine_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Объём двигателя:
    - запятые → точки, убрать L и суффиксы TFSI/TDI/Turbo/…;
    - фикс мусорной запятой в конце ('4,3,' → 4.3);
    - Niva: 21.6..2.7 → 1.7;
    - Jaguar: 400 → 4.0;
    - мусор > 50 → NaN (логируется).
    """
    lost = []

    def _extract(row):
        x = row["EngineVolume"]
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace("L", "").replace("l", "")
        s = re.sub(
            r"\s*(MPI|GDI|CRDI|TDI|Turbo|TFSI|TDCI|DI|VGT|WGT|Common Rail|T-GDI)\s*",
            "", s, flags=re.IGNORECASE,
        )
        m = re.search(r"([\d.,]+)", s)
        if not m:
            lost.append((x, "no digit"))
            return np.nan
        raw_num = m.group(1).replace(",", ".").rstrip(".")
        try:
            val = float(raw_num)
        except ValueError:
            lost.append((x, "float parse"))
            return np.nan

        model = str(row.get("Model", "")).lower()
        if "niva" in model and 2.0 < val < 30.0:
            return 1.7
        if val >= 100:
            return round(val / 100.0, 1)
        if val > 50:
            lost.append((x, "val>50"))
            return np.nan
        return val

    before_na = df["EngineVolume"].isna().sum()
    df["EngineVolume"] = df.apply(_extract, axis=1)
    after_na = df["EngineVolume"].isna().sum()
    print(f"EngineVolume: NaN было {before_na}, стало {after_na} "
          f"(потеряно {after_na - before_na})")
    if lost:
        print(f"  Примеры потерянных значений ({len(lost)}):", lost[:10])
    return df


# ---------- 4. Компания и регионы ----------
def normalize_company(df: pd.DataFrame) -> pd.DataFrame:
    """Неявные дубликаты названий дилерских центров."""
    def _norm(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if "mercur" in s or "меркур" in s:
            return "Mercur Auto"
        if "бавария" in s or "bavaria" in s:
            return "AvtoTsentr-Bavaria"
        if "autokapital" in s or "автокапитал" in s:
            return "Autokapital"
        if "ммс рус" in s or "mmc rus" in s:
            return "MMC Rus"
        if "равон" in s or "ravon" in s:
            return "Ravon Motors Kazakhstan"
        if "caspian" in s or "каспиан" in s:
            return "Caspian Motors"
        return str(x).strip()

    df["Company"] = df["Company"].apply(_norm)
    print("Company:\n", df["Company"].value_counts().to_string())
    return df


def normalize_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Регистр: каждое слово с заглавной, 'область' — строчная."""
    def _cap_word(w: str) -> str:
        return "-".join(part.capitalize() for part in w.split("-"))

    def _norm(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        if not s:
            return s
        out = []
        for w in s.split():
            lw = w.lower()
            if lw == "область":
                out.append("область")
            elif lw in {"г.", "город"}:
                out.append(w.capitalize())
            else:
                out.append(_cap_word(w))
        return " ".join(out)

    df["Region"] = df["Region"].apply(_norm)
    df["Region2"] = df["Region2"].apply(_norm)
    return df


# ---------- 5. Дата продажи (ДО агрегации) ----------
def create_sale_date(df: pd.DataFrame) -> pd.DataFrame:
    """SaleDate = последний день месяца (в исходнике только год и месяц)."""
    month_map = {
        "Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4,
        "Май": 5, "Июнь": 6, "Июль": 7, "Август": 8,
        "Сентябрь": 9, "Октябрь": 10, "Ноябрь": 11, "Декабрь": 12,
    }
    df["MonthNum"] = df["Month"].map(month_map).astype("Int64")
    df["SaleDate"] = (
        pd.to_datetime(
            df["Year"].astype("Int64").astype(str) + "-"
            + df["MonthNum"].astype("Int64").astype(str) + "-01",
            errors="coerce",
        )
        + pd.offsets.MonthEnd(0)
    )
    bad = df["SaleDate"].isna().sum()
    print(f"SaleDate: NaT {bad} (Year/Month не распарсились)")
    df = df.drop(columns=["Year", "Month", "MonthNum"])
    return df


# ---------- 6. Флаги аномалий (ДО агрегации) ----------
def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Флаги аномалий ставятся ДО агрегации, чтобы не потерять
    возвраты и оптовые заказы при схлопывании строк.
    """
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["SaleAmount"] = pd.to_numeric(df["SaleAmount"], errors="coerce")

    df["is_return"] = (df["Quantity"] < 0).astype("int8")
    print(f"Возвратов (Quantity<0): {df['is_return'].sum()}")

    # Эвристика: 10+ единиц в одной строке — предположительно опт
    df["is_wholesale"] = (df["Quantity"] >= 10).astype("int8")
    print(f"Оптовых (Quantity>=10): {df['is_wholesale'].sum()}")

    # Мусорные строки: нулевая выручка и нет количества
    bad_zero = (df["SaleAmount"] == 0) & (df["Quantity"].isna())
    if bad_zero.any():
        print(f"Удалено строк SaleAmount=0 & Quantity=NaN: {bad_zero.sum()}")
        df = df[~bad_zero]
    return df


# ---------- 7. Агрегация ----------
def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Схлопывание дублей агрегацией:
    Quantity и SaleAmount суммируются, Price — средняя,
    флаги — max (чтобы не потерять возврат/опт).
    ВАЖНО: drop_duplicates() здесь НЕ используется —
    он удаляет строки и теряет ~44% Quantity и ~50% SaleAmount.
    """
    before = len(df)
    for col in ["Quantity", "Price", "SaleAmount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(GROUP_COLS, dropna=False, as_index=False)
          .agg(
              Quantity=("Quantity", "sum"),
              Price=("Price", "mean"),
              SaleAmount=("SaleAmount", "sum"),
              is_return=("is_return", "max"),
              is_wholesale=("is_wholesale", "max"),
          )
    )
    print(f"Строк до агрегации: {before}")
    print(f"Строк после агрегации: {len(grouped)}")
    print(f"Схлопнуто дублей: {before - len(grouped)}")
    print(f"Quantity: {df['Quantity'].sum():.0f} -> "
          f"{grouped['Quantity'].sum():.0f}")
    print(f"SaleAmount: {df['SaleAmount'].sum():.2f} -> "
          f"{grouped['SaleAmount'].sum():.2f}")
    return grouped


# ---------- 8. Финальные типы ----------
def finalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """SaleDate → datetime, ProductionYear → Int64, категориальные → category."""
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    df["ProductionYear"] = pd.to_numeric(
        df["ProductionYear"], errors="coerce"
    ).astype("Int64")
    for col in ["FuelType", "Transmission", "DriveType",
                "Segment2013", "Class2013"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("\n=== DTYPES ===")
    print(df.dtypes.to_string())
    return df


# ---------- 9. Контроль ----------
def report_control(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """Суммы до/после, пропуски, потери по брендам/моделям."""
    print("\n=== КОНТРОЛЬ ===")
    print(f"Строк до очистки : {len(df_before)}")
    print(f"Строк после       : {len(df_after)}")

    print("\nСуммы до/после:")
    print(f"  Quantity   : {df_before['Quantity'].sum():.0f} -> "
          f"{df_after['Quantity'].sum():.0f}")
    print(f"  SaleAmount : {df_before['SaleAmount'].sum():.2f} -> "
          f"{df_after['SaleAmount'].sum():.2f}")
    print(f"  Price(mean): {df_before['Price'].mean():.2f} -> "
          f"{df_after['Price'].mean():.2f}")
    print("  (Price mean меняется из-за смены весов при агрегации — это норма)")

    print("\nПропуски после:")
    na = df_after.isna().sum()
    print(na[na > 0].sort_values(ascending=False).to_string())

    print("\n=== ПОТЕРИ ПО БРЕНДАМ/МОДЕЛЯМ (топ по lost_%) ===")
    o = df_before.copy()
    o["Q"] = pd.to_numeric(o["Quantity"], errors="coerce")
    c = df_after.copy()
    c["Q"] = pd.to_numeric(c["Quantity"], errors="coerce")
    o_g = o.groupby(["Brand", "Model"], observed=True)["Q"].sum().rename("ORIG_Q")
    c_g = c.groupby(["Brand", "Model"], observed=True)["Q"].sum().rename("CLEAN_Q")
    cmp = pd.concat([o_g, c_g], axis=1).fillna(0)
    cmp = cmp[cmp["ORIG_Q"] > 0]
    cmp["diff"] = cmp["CLEAN_Q"] - cmp["ORIG_Q"]
    cmp["lost_%"] = (cmp["diff"] / cmp["ORIG_Q"] * 100).round(1)
    print(cmp.sort_values("lost_%").head(15).to_string())


def main():
    df_raw = load_raw()

    # Снимок «до» для контроля сумм
    df_before = df_raw.copy()
    for col in ["Quantity", "Price", "SaleAmount"]:
        df_before[col] = pd.to_numeric(df_before[col], errors="coerce")

    # 1. Удаление ненужных полей — строго по ТЗ
    df = drop_useless_columns(df_raw)

    # 2. Нормализация категорий и аномалий
    df = encode_country(df)
    df = encode_fuel(df)
    df = normalize_drive(df)
    df = normalize_transmission(df)
    df = clean_engine_volume(df)
    df = normalize_company(df)
    df = normalize_regions(df)

    # 3. Дата продажи и флаги — ДО агрегации
    df = create_sale_date(df)
    df = add_flags(df)

    # 4. Агрегация дублей (сохраняет объём и выручку)
    df = aggregate_duplicates(df)

    # 5. Финальные типы
    df = finalize_types(df)

    # 6. Контроль
    report_control(df_before, df)

    # 7. Сохранение
    df.to_csv("autokz2019_cleaned.csv", sep=";", index=False, encoding="utf-8-sig")
    print("\nСохранено: autokz2019_cleaned.csv")


if __name__ == "__main__":
    main()