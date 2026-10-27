# Iceberg regression fixtures

The equality-delete dataset and `../catalog_fixture.py` originate from
[AstroVela/duckdb-iceberg](https://github.com/AstroVela/duckdb-iceberg/tree/8ef07c9),
commit `8ef07c9` (the pre-FTS base). The MIT notice is retained in
[LICENSE.duckdb-iceberg](LICENSE.duckdb-iceberg).

The tests run from this fixture directory so embedded relative Avro paths resolve.
No separate repository checkout or companion PR is required.
