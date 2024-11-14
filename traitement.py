import pandas as pd

df = pd.read_csv("./data/x_train.csv").dropna()

for i in range(df["SECTOR"].unique().max()+1):
	dfi = df[df["SECTOR"] == i]
	dfi.to_csv("./data/sector_" + str(i) + ".csv")
