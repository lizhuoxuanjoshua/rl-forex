import pandas as pd

df=pd.read_csv("重构后999.csv")
df=df[-len(df)//3:].reset_index(drop=True)
df.to_csv("data.csv")
print((df.sum()))
