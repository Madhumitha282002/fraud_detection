from feast import Entity

transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    description="Unique transaction identifier",
)