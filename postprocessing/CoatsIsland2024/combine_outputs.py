import pandas as pd
import os

# Ange sökvägen till mappen som innehåller dina CSV-filer
mapp_path = '/Volumes/T7 Shield/Analysis/Hudson Bay 2024/Run 1/csv'

# Hämta en lista med alla CSV-filer i mappen
filenames = [f for f in os.listdir(mapp_path) if f.endswith('.csv')]

# Skapa en tom lista för att lagra dataframes
df_list = []

# Läs in varje CSV-fil och lägg till dataframe till listan
for filename in filenames:
    file_path = os.path.join(mapp_path, filename)
    df = pd.read_csv(file_path)
    df_list.append(df)

# Slå samman alla dataframes till en
combined_df = pd.concat(df_list, ignore_index=True)

# Spara den sammanslagna dataframen till en ny CSV-fil
combined_df.to_csv('/Volumes/T7 Shield/Analysis/Hudson Bay 2024/Run 1/combined.csv', index=False)

print("Sammanslagningen är klar och filen är sparad.")