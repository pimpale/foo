#%%
import duckdb

src1 = duckdb.read_csv("./data_models/trials/axel.csv")
src2 = duckdb.read_csv("./data_models/trials/govind.csv")

joint = duckdb.sql("SELECT * FROM src1 UNION ALL SELECT * FROM src2").df()
joint.to_csv("./data_models/trials/joint.csv", index=False)