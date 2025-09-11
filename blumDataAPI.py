from datasets import load_dataset

# Dataset direkt vom Hugging Face Hub laden
ds = load_dataset("danidanou/Bloomberg_Financial_News", split="train")

# Nur die ersten 1000 Zeilen auswählen
ds_small = ds.select(range(5000))

# Beispiel ausgeben
print(ds_small[0])
print("Anzahl Artikel:", len(ds_small))

# Wenn du mit Pandas arbeitest
import pandas as pd

df = ds_small.to_pandas()
df.to_csv("bloomberg_news_1000.csv", index=False)
