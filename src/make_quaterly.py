import pandas as pd

# === 1. Загрузка исходных данных ===
file_path = "../data/merged_df_readable.csv"   # замени на свой путь
df = pd.read_csv(file_path)

# === 2. Преобразуем даты ===
df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
print("\nКоличество некорректных дат:", df['TIME_PERIOD'].isna().sum())

df = df.set_index('TIME_PERIOD').sort_index()
print("\nТип индекса:", type(df.index))

# === 3. Определяем частоту колонок ===
monthly_cols = [
    'Inflation (HICP Manufacturing)',
    'Unemployment Rate',
    'Air Passenger Transport',
    'Industrial Production Index',
    'Retail Trade Turnover',
    'Tourist Overnight Stays'
]

quarterly_cols = [
    'GDP (Quarterly)',
    'Employment'
]

semiannual_cols = ['Energy Prices']

annual_cols = [
    'Exports (National Accounts)',
    'Emigration of Citizens',
    'Road Passenger Transport',
    'Freight Transport',
    'Net Migration (World Bank)',
    'Population'
]

# === 4. Смещаем годовые данные (мягкий способ без freq) ===
for col in annual_cols:
    if col in df.columns:
        # Преобразуем индекс в конец года вручную
        df[col] = df[col].copy()
        df[col].index = df.index + pd.offsets.YearEnd(0)

# === 5. Создаём квартальную сетку ===
df_quarterly = pd.DataFrame(index=pd.date_range(
    start=df.index.min(), end=df.index.max(), freq='QE'))

# === 6. Обработка по типам частоты ===

# Месячные → средние за квартал
for col in monthly_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').mean()
        df_quarterly[col] = temp[col]

# Квартальные → просто last
for col in quarterly_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').last()
        df_quarterly[col] = temp[col]

# Полугодовые → заполняем вперед
for col in semiannual_cols:
    if col in df.columns:
        temp = df[[col]].resample('QE').ffill()
        df_quarterly[col] = temp[col]

# Годовые → расширяем значения по кварталам
for col in annual_cols:
    if col in df.columns:
        temp = df[[col]].resample('YE').last()  # конец года
        temp = temp.resample('QE').ffill()       # на кварталы
        df_quarterly[col] = temp[col]

# === 7. Отбираем только значения начиная с 2000 года ===
df_quarterly = df_quarterly[df_quarterly.index >= '2000-01-01']

# === 8. Удаляем полностью пустые столбцы ===
df_quarterly = df_quarterly.dropna(axis=1, how='all')

# === 9. Финальная очистка и сохранение ===
df_quarterly = df_quarterly.sort_index().ffill().bfill()
df_quarterly.to_csv("../data/merged_quarterly.csv", index_label="TIME_PERIOD")

print("\nSUCCESS: Данные приведены к квартальной частоте и сохранены в '../data/merged_quarterly.csv'")
print("Размерность:", df_quarterly.shape)
print("Период:", df_quarterly.index.min(), "→", df_quarterly.index.max())