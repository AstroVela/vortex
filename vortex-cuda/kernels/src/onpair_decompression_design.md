# OnPair GPU decompression design

## Representation facts

The native Vortex OnPair dictionary offset type is 32-bit, not 64-bit:
`OnPairDictionaryStorage::offsets` is `Buffer<u32>` and implements
`DictionaryStorage<u32>` in `encodings/onpair/src/array.rs`.

The optimized CUDA decoder does not read that offset directory for every token.
CPU staging flattens each dictionary entry into cache-oriented planes:

- `dict_s8_lo`: bytes 0 through 7, zero-padded, 8 bytes per entry.
- `dict_s8_hi`: bytes 8 through 15, zero-padded, 8 bytes per entry.
- `packed_lens`: two four-bit encoded lengths per byte.

The code stream remains native `u16`, or 2 bytes per token. It is a large
streaming input and is not expected to remain resident in L1. The three
dictionary tables are the reusable, cache-sized working set.

## Packed length encoding

OnPair dictionary entries have lengths in `[1, 16]`. A raw nibble cannot
represent 16, so the host stores `length - 1`:

```text
even code: packed_lens[code / 2] bits 3:0
odd code:  packed_lens[code / 2] bits 7:4
decoded:   nibble + 1
```

The device helper is:

```cpp
__device__ inline uint32_t unpack_length(
    const uint8_t *__restrict packed_lens, uint32_t code) {
    const uint32_t packed = (uint32_t)packed_lens[code >> 1u];
    const uint32_t shift = (code & 1u) << 2u;
    return ((packed >> shift) & 0xfu) + 1u;
}
```

This handles the full range without a sentinel or side table. Tokens past the
end of the stream still receive length zero in registers and do not read the
packed table.

## Kernel ABI and loads

`onpair_decompress` receives:

```cpp
const uint16_t *codes;
const uint64_t *chunk_offsets;
const uint8_t *dict_s8_lo;
const uint8_t *dict_s8_hi;
const uint8_t *packed_lens;
uint8_t *output_bytes;
uint64_t total_tokens;
```

Every in-range token performs one 8-byte low-plane load and one packed-length
byte load. Only entries longer than eight bytes enter the existing dense
per-warp request queue and perform an 8-byte high-plane load. The scan, shared
staging, aligned output drain, and 128-token-per-warp assignment are unchanged.

## Working-set sizes

For `N` dictionary entries:

| table | previous split decoder | packed split decoder |
|---|---:|---:|
| low bytes | `8N` | `8N` |
| high-byte source | `16N` padded dictionary | `8N` high plane |
| lengths | `N` | `ceil(N / 2)` |
| total GPU dictionary staging | `25N` | `16.5N` |

At 4096 entries this is 102,400 bytes previously versus 67,584 bytes now:
32 KiB low + 32 KiB high + 2 KiB lengths, a 34.0% reduction. The length table
itself is exactly halved, and the high-byte source is halved.

The benchmark metadata records the actual per-cell values as `code_bytes`,
`dict_s8_lo_bytes`, `dict_s8_hi_bytes`, and `packed_lens_bytes`.

## Correctness constraints

- Host staging rejects dictionary lengths outside `[1, 16]`.
- Low and high planes are 8-byte-strided and loaded as aligned `uint2`.
- High bytes are read only when decoded length exceeds eight.
- The high request stores `high_length - 1`, which fits in three bits for
  high lengths `[1, 8]`.
- GPU output is copied back and compared byte-for-byte with CPU decode for
  every benchmark process.

## Source files

- `onpair_decompress.cu`: packed-length, low/high-plane candidate.
- `onpair_decompress_u8_lens.cu`: preserved previous split candidate.
- `onpair_old_2.cu`: preserved legacy comparison kernel.

## Shared-memory drain capacity and handoff

For the six-token kernel, each warp owns 192 tokens. Static shared memory has
two independent consumers:

~~~text
drain staging = 8 warps * (192 * bytes_per_token + 32 alignment bytes)
high requests = 8 warps * 192 requests * 4 bytes
~~~

The original 16-byte worst-case drain uses 24,832 bytes for staging and 6,144
bytes for high requests, or 30,976 bytes/block. The compact candidate sets
`bytes_per_token = 12`, reducing staging to 18,688 bytes and total static shared
memory to 24,832 bytes/block. The launch bound then makes ptxas allocate 48
rather than 64 registers/thread with zero spills.

The compact size is not a format restriction. After the warp prefix scan, the
kernel compares `warp_total` with the 2,304-byte payload capacity. Warps that
fit retain the existing dense high-request queue and coalesced shared-memory
drain. An overflowing warp writes complete low/high token ranges directly to
global memory and returns, so every legal 1-16-byte dictionary distribution
remains correct.

### Partial high-drain experiment

A more aggressive experiment staged only the first eight bytes/token (1,536
payload bytes/warp) and sent the suffix directly to global memory. Only high
requests whose destination began inside that prefix were queued. At most
`floor(1536 / 9) = 170` such requests can exist because every high-bearing
token consumes at least nine output bytes. This reduced shared memory to:

~~~text
drain staging = 8 * (1536 + 32) = 12,544 bytes
high requests = 8 * 170 * 4    =  5,440 bytes
total                                  17,984 bytes/block
~~~

The result compiled to 48 registers with zero spills and validated byte-exactly,
but it was slower: 1,189.0 GB/s on TPC-H `l_comment` and 927.8 GB/s on
ClickBench `URL`. Per-byte routing was much worse (887.2 GB/s on `l_comment`);
token-range routing recovered most of that loss but remained behind the
12-byte compact kernel. The smaller allocation could support seven blocks by
shared-memory capacity, but 48 registers still limit residency to five blocks,
so direct/scattered stores and routing add cost without increasing occupancy.

Another experiment reused the high-request queue as temporary destination
scratch. It reached 40 registers, 21,760 bytes shared, zero spills, and 70.64%
achieved occupancy. It nevertheless lost performance: L1 hit rate fell from
85.36% to 80.54%, L2 read sectors rose from 18.8M to 33.6M, and long-scoreboard
stall rose to 22.61%. Reordering low emission ahead of high compaction removed
the baseline's first-high-load latency overlap and damaged dense cache reuse.

The next useful optimization must preserve all three properties of the current
fast path: six-token grid amortization, plane-major dense high requests, and
the first high load overlapping low-byte emission. Merely reducing shared
memory below 24,832 bytes does not help while the 48-register allocation
already limits the kernel to five resident blocks.

## Adaptive nine-byte drain

Most measured code occurrences decode to at most eight bytes, but retaining all
six per-thread low-plane values keeps twelve 32-bit words live. Reducing the
shared drain alone is insufficient if those values keep register residency at
five blocks/SM. The adaptive kernel combines two changes:

- allocate nine output bytes/token, or 1,728 payload bytes per 192-token warp;
- keep only the first `uint2` low value live across the prefix scan and dense
  high gather, then reload the other five low values while draining.

Keeping one low value preserves independent work after the first dense
high-plane load. Reloading the remaining five short values costs global-load
instructions, but their 32 KiB plane is reusable and cache-resident; shortening
their live ranges reduces the compiled allocation from 48 to 40
registers/thread. The dense high-request queue and the first-high-load overlap
are unchanged.

Static shared memory becomes:

~~~text
drain staging = 8 * (192 * 9 + 32) = 14,080 bytes
high requests = 8 * 192 * 4        =  6,144 bytes
total                                      20,224 bytes/block
~~~

On sm_90, `onpair_decompress_6tpt_cap9_keep1_lb6` compiles with 40 registers,
20,224 bytes static shared memory, and no spills. Both registers and shared
memory permit six blocks/SM, versus five for the 48-register, 24,832-byte
cap-12 kernel.

This kernel deliberately has no device overflow path: compiling the fallback
back in raises register pressure and loses the residency gain. The caller must
compute the maximum adjacent `chunk_offsets` difference for the same 192-token
chunking and use cap-9 only when it is at most 1,728. Otherwise it must launch
`onpair_decompress_6tpt_cap12_lb5`. The offsets are already produced during
host staging, so this is a whole-column selection and does not change or
annotate individual codes.

The code stream remains the original two-byte `u16` sequence. Lengths still
come from the separate packed four-bit table. No code bits, dictionary entries,
or serialized data are changed.
