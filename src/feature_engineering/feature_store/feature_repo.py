from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.data_format import ParquetFormat

from entities import transaction

transaction_source = FileSource(
    name="transaction_source",
    path="../../../data/processed/transactions_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format=ParquetFormat(),
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=None,
    schema=[
        Field(name="amount_log", dtype=Float64),
        Field(name="amount_zscore", dtype=Float64),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="tx_frequency_1h", dtype=Int64),
        Field(name="tx_frequency_6h", dtype=Int64),
        Field(name="tx_frequency_24h", dtype=Int64),
        Field(name="amount_mean_1h", dtype=Float64),
        Field(name="amount_mean_24h", dtype=Float64),
        Field(name="time_since_last_tx", dtype=Float64),
    ],
    source=transaction_source,
    online=True,
)
