import mltable
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()
ml_client = MLClient.from_config(credential=credential)
data_asset = ml_client.data.get("sosnovski_20250518_121624", version="1")

tbl = mltable.load(data_asset.path)

df = tbl.to_pandas_dataframe()
df