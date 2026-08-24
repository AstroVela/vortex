# Convert

If you haven't already, download the sample data (see [Install](install.md#sample-data)):

```bash
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

The `vx convert` command converts a Parquet file to Vortex, applying compression automatically:

```bash
vx convert yellow_tripdata_2024-01.parquet
```

This produces `yellow_tripdata_2024-01.vortex` in the same directory. By default it uses
BtrBlocks compression, chunking on Parquet row-group boundaries.

## Compression strategies

Choose a compression strategy with `--strategy`:

```bash
# Default: BtrBlocks compressor
vx convert yellow_tripdata_2024-01.parquet --strategy btrblocks

# Compact: more aggressive compression
vx convert yellow_tripdata_2024-01.parquet --strategy compact
```

## Converting from object storage

The input can be a URL instead of a local path, in which case `vx convert` reads the Parquet
file directly from the store — only the Vortex output is written to disk:

```bash
# S3, GCS, Azure, or plain HTTP
vx convert s3://bucket/data/events.parquet --output events.vortex

# A Hugging Face dataset
vx convert hf://datasets/openai/gsm8k/main/test-00000-of-00001.parquet
```

Without `--output`, the file is named after the last path segment, so the second command above
writes `test-00000-of-00001.vortex` in the working directory.

URLs resolve through the same registry every other Vortex binding uses, so credentials come from
the usual environment variables — `AWS_*` for `s3://`, `HF_TOKEN` for a private or gated `hf://`
repository, and so on.

### Hugging Face revisions

`hf://` URLs take an optional revision, which is how you reach a dataset's auto-converted Parquet
branch. A revision containing `/` must be percent-encoded:

```bash
vx convert 'hf://datasets/Anthropic/hh-rlhf@refs%2Fconvert%2Fparquet/default/test/0000.parquet'
```

Prefer `hf://` over the equivalent `https://huggingface.co/...` URL. The HTTP store
percent-decodes the path, so an encoded revision like the one above resolves to a path the Hub
does not serve; the `hf://` scheme encodes it correctly.
