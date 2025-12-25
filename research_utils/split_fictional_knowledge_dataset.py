import pandas as pd

df = pd.read_json(r"fictional_knowledge\\fictional_knowledge.json")
#%%

train_data=df[["train_context"]].rename(columns={"train_context":"text"})
#%%

train_data.to_parquet(r"fictional_knowledge\\train_data.parquet")