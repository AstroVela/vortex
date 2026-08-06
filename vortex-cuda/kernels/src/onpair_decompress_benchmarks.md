# OnPair CUDA decompression validation

Date: 2026-08-06  
GPU: NVIDIA GH200 480GB  
Measured algorithm: the baseline now retained as `onpair_old_2.cu` /
`onpair_old_2`. During this run the same baseline body was temporarily exported
from `onpair_decompress.cu` as `onpair_decompress`.

> These numbers do not measure the restored best universal kernel currently in
> `onpair_decompress.cu`. They are retained as the baseline result for the next
> direct A/B run.

## Input contract

The kernel consumes the original flat `u16` OnPair code stream. It does not
repack or otherwise change the codes. The CPU supplies the uniform dictionary
views (`dict_s8`, 16-byte-padded dictionary, and `u8` lengths). The output
position of each 128-token warp batch is already known in `chunk_offsets`, so
these measurements time only dictionary decode and output writes.

## Method

- Release CUDA build (`nvcc -O3`, native GPU architecture).
- Eight warps per block; each warp handles 128 codes at four codes per lane.
- Two untimed warmups followed by 100 CUDA-event-timed iterations.
- Five fresh processes per column; table reports the median and full range.
- `CUDA_MODULE_LOADING=EAGER` and `ONPAIR_FAST=1`.
- No other GPU benchmarks ran concurrently.
- Correctness was checked separately for every column by copying the output
  back and comparing every byte with CPU decode (50 timed iterations).
- Seven inputs decode approximately 1.0 GB. TPC-H `l_shipmode` contains only
  257 MB at SF10, so the same saved chunk was supplied four times to decode
  1.028 GB and avoid a small-input result.

`GB/s` below uses decimal bytes. `GiB/s` is also included so results match the
benchmark JSON. `H2D + decode` estimates copying the stored compressed payload
to the GPU followed by this kernel; it is not part of the kernel-only timing.

## Results

| Dataset / column | Decoded GB | Stored input GB | Median ms | Min–max ms | Kernel GiB/s | Kernel GB/s | H2D + decode GB/s | Verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| TPC-H `l_comment` | 1.000000 | 0.255905 | 2.335821 | 2.317003–2.336501 | 398.713 | 428.115 | 293.794 | yes |
| TPC-H `l_shipmode` (4 chunks) | 1.028347 | 0.479891 | 0.891137 | 0.887951–0.891770 | 1074.720 | 1153.972 | 336.436 | yes |
| ClickBench `URL` | 1.000000 | 0.428147 | 1.909591 | 1.772180–1.909828 | 487.708 | 523.672 | 270.408 | yes |
| FineWeb `text` | 0.999999 | 0.594903 | 2.786300 | 2.782168–2.786543 | 334.250 | 358.898 | 187.663 | yes |
| GH Archive `event_json` | 1.000000 | 0.448999 | 1.747309 | 1.745585–1.775309 | 533.004 | 572.308 | 275.087 | yes |
| HDFS `log_line` | 1.000000 | 0.302199 | 1.380700 | 1.336167–1.381106 | 674.529 | 724.270 | 375.920 | yes |
| CodeSearchNet `func_code_string` | 0.999996 | 0.532221 | 1.949341 | 1.788100–1.949874 | 477.761 | 512.992 | 238.185 | yes |
| FineWeb2 Chinese `text` | 1.000000 | 0.632165 | 2.988162 | 2.987273–2.989148 | 311.671 | 334.654 | 176.872 | yes |

## Interpretation

The previous greater-than-1-TB/s result is reproducible for `l_shipmode`, even
after increasing the timed decode to 1.028 GB. It is a favorable workload:
every dictionary token is at most eight bytes, so the conditional high-half
dictionary gather is never taken. It should not be treated as the universal
kernel rate.

Across the diverse 1 GB inputs, kernel-only throughput ranges from 334.7 GB/s
for multibyte FineWeb2 text to 724.3 GB/s for HDFS logs, with `l_shipmode` as
the 1.154-TB/s outlier. Seven of eight cases remain below the desired 800 GB/s.
