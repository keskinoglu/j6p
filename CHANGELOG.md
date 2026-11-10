## v0.2.0 (2026-11-10)

### Feat

- **extractors.py,-loaders.py**: add internal_counterparties_txt_reader as an extractor and csv_writer as a loader
- **extractors.py**: add schwab_banking_csv_reader as an extractor
- **transformers.py**: add schwab_banking_fuser
- **transformers.py**: add citi transformers plus handle potential duplicate rows in other accounts as well
- **extractors.py**: add citi_banking_csv_reader
- **transformers.py**: add ing_de_fuser
- **transformers.py**: add ing_de_transformer
- **extractors.py**: add ing_de_csv_reader
- **transformers.py**: add wise fuser
- **transformers.py**: add wise transformer
- **extractors.py,-Makefile**: add wise csv extractor plus modify the make lf command to make fix
- **transformers.py**: add schwab_banking_transformer
- **src/**: update the main datatype from polars native lazyframe to a wrapper class called AnnotatedLazyFrame which bundles the lazyframe along with a metadat map
- **transformers.py**: add schwab_brokerage_fuser
- **transformers.py**: add steps to the schwab_brokerage_transformer to convert numerical string values to numerical types
- **transformers.py**: add transformation steps for date column of the schwab brokerage source
- **blocks.py,-fusers.py,-readers.py,-transformer.py,-writers.py**: add initial data pipeline design
- **wiring**: add pyproject.toml, uv.lock, and makefile

### Fix

- **calculate-balances-by-using-the-most-recent-known-account-balance-and-assuming-it-continues-back-in-time-until-an-older-known-account-balance-contradicts-the-current-running-balance**: The newest values are consider more truthful so any discrepencies (which would come from missing data for example) are handeld by accepting the newest known account balance as truth until a contradiction occurs with an older known account balance. At that time, the older known account balance is taken to be correct and continues back in time until the end is reached. If the end is reached while there is balance remaining, the balance is assumed to be zero. Transactions are added at these points to correct the running balance
- **transformers.py,-extractors.py**: add multi-bank fuser and known-balance ETL for sources whose transaction exports don't contain running balances
- **loaders.py,-pyproject.toml**: add cslx_writer as a new loader
- **transformers.py**: maintain order during sorting to keep intra-day transactions in order
- **transformers.py**: stip commas in _convert_usd_string_to_decimal which was formerly dropping values over one thousand

### Refactor

- **extractors.py**: rename schwab_json_reader signature for clarity
- **etl.py,-extractors.py,-loaders.py,-transformers.py,-type_aliases.py,-test_etl.py**: create symmetry between function generators and rename Writer -> Loader and Reader -> Extractor
- **etl.py,-test_etl.py**: rename block to etl (both filename and class)
- **codebase**: convert frame references to lazy_frame references for improved clarity
- **transformers.py,-loaders.py,-extractors.py**: fold fusers.py into transformers.py and convert readers.py to  extractors.py, writers.py to loaders.py
- **readers.py,-writers.py**: update signatures
- **blocks.py,-type_aliases.py**: refactor blocks.py to adhere to SOLID coding principles
- **main.py**: remove main.py
