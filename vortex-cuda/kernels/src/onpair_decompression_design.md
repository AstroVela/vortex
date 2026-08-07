# OnPair CUDA decoder: baseline to best universal kernel

This note explains every intentional algorithmic difference between the two
retained CUDA translation units:

- `onpair_old_2.cu` / `onpair_old_2`: the loadable baseline.
- `onpair_decompress.cu` / `onpair_decompress`: the best universal kernel from
  the original-code-layout experiments.

The old kernel remains so benchmarks can make a direct, reproducible A/B
comparison. The new kernel is the production candidate.

## Fixed input contract

Neither kernel changes the OnPair code representation:

- Codes are the original flat `u16` stream produced by OnPair. No length bits,
  alternate ordering, or other metadata are packed into the codes.
- A warp owns 128 consecutive codes. Lane `L` reads code `L + 32*k` for
  `k = 0..3`, so each plane is a naturally coalesced streaming read.
- The CPU supplies a uniform dictionary representation: an 8-byte low table,
  a 16-byte-padded full table, and one `u8` length per dictionary entry.
- `chunk_offsets[chunk]` supplies the persisted output base for each 128-token
  warp batch. The kernel still calculates token offsets within that batch.
- Inputs and output are resident on the GPU for kernel-only timing.

An ordinary `codes[i]` load is retained deliberately. The SASS investigation
showed it compiling through the normal read-only/non-coherent global path.
Explicit `.cg` and `.cs` code-load policies were tested but did not establish a
universal win. The new kernel therefore improves scheduling without changing
the code stream or requiring a second code buffer.

### Code streaming versus dictionary caching

The codes and dictionary tables have different reuse patterns:

- Each `u16` code is consumed exactly once, in four coalesced 32-lane planes.
  It is therefore logically streaming data.
- The 4,096-entry `dict_s8` table is 32 KiB and the `lens` table is 4 KiB.
  Their indices are data-dependent, but the same 36 KiB working set is reused
  by every warp. Keeping useful lines in cache can therefore avoid repeated L2
  or device-memory latency.

It is tempting to force code loads to `.cg` (bypass L1) or `.cs` (evict first)
to protect the reusable dictionary lines. Direct tests did not justify doing
so in the universal kernel. On ClickBench URL, ordinary / `.cg` / `.cs` code
loads measured 2.432 / 2.456 / 2.465 ms. On `l_comment` they measured 1.646 /
1.633 / 1.643 ms. Thus ordinary loads won URL by about 0.9%, `.cg` won
`l_comment` by about 0.8%, and `.cs` won neither. With `.cg`, the aggregate L1
hit rate increased from 91.510% to 92.506% and L1 misses fell by 5.38 million,
but L2 misses were unchanged and long-scoreboard stalls increased from 14.451%
to 16.076%. Protecting L1 residency did not shorten the critical dependency
chain reliably.

The lower dictionary is also not copied into shared memory. CUDA provides no
portable launch option that pins an arbitrary global allocation in L1, and a
per-block copy would duplicate work and require barriers. The measured
lens-only shared-memory experiment was 36.6% slower on URL; persistent
prefetch was 33.3% slower. The retained policy is consequently: ordinary
coalesced loads for the one-pass codes, ordinary cached gathers for `dict_s8`
and `lens`, and instruction scheduling to overlap their exposed latency.

## The four decompression parts

### 1. Read code, length, and dictionary value

Baseline:

1. Each lane reads four `u16` codes in four warp-wide planes.
2. Each code immediately selects its `u8` length and an unconditional `uint2`
   containing the first eight dictionary bytes.
3. The token-owning lane later performs its own conditional `uint2` high-half
   load when `length > 8`.

Best kernel:

1. The code, length, and unconditional low-eight-byte reads are unchanged.
2. High-half reads are deferred and issued through the dense request stream
   described under part 3.

Why retain the split dictionary read: most tokens are short, so an
unconditional 16-byte dictionary load wastes bandwidth and cache requests.
The low table is 32 KiB for a 4,096-entry dictionary; the padded table is only
touched for tokens longer than eight bytes. This also keeps four `uint2`
values rather than four `uint4` values live per lane.

Why optimize this dependency chain: phase ablation measured load plus cursor
work at 61% of full URL time and 55% of full `l_comment` time. It is a latency
and issue-shape bottleneck, not simply a peak-bandwidth problem.

### 2. Calculate and update the token cursor

Baseline:

- Four independent 32-bit warp scans calculate the four token planes.
- This requires 20 `SHFL.UP` operations plus four lane-31 total broadcasts.
- Plane totals are chained to form the batch-relative exclusive cursor for
  every token.

Best kernel:

- The four lengths are packed into one logical `u64` at bit positions
  0, 16, 32, and 48.
- One logical 64-bit warp scan calculates all four prefixes together.
- On the GPU this is ten 32-bit `SHFL.UP` operations plus two broadcasts,
  halving the serial shuffle chain.
- The four fields are extracted and their plane totals are chained exactly as
  in the baseline, so token order and destinations do not change.

Why packed addition is exact: a plane contains 32 tokens and an OnPair token
is at most 16 bytes, so any plane prefix is at most 512. It cannot overflow a
16-bit field or carry into the adjacent plane.

Controlled measurements found a 4.1-5.3% kernel-time reduction across ten
columns. On URL, executed instructions fell 4.3% and MIO-throttle fell 27%,
while global-load sectors, global-store sectors, and shared conflicts remained
effectively unchanged. That isolates the improvement to cursor instructions
and their dependency chain.

### 3. Write dictionary values into the staging buffer

Baseline:

- Every token owner writes its low bytes to its exclusive shared-memory range.
- Owners with `length > 8` independently load and write their high bytes.
- The high-half gather therefore runs with a sparse, divergent set of lanes
  and at four different points in the per-thread token loop.

Best kernel:

1. Each token plane uses `ballot` to identify long tokens.
2. `popc` assigns each long token a dense rank across all four planes.
3. The owner appends one packed request to a per-warp shared queue. A request
   contains the 16-bit code, 12-bit shared destination, and three-bit
   `high_length - 1`.
4. After a warp synchronization, consecutive active lanes consume consecutive
   requests and issue the high-half dictionary gathers.
5. The first dense high-half load is issued before the independent low-byte
   shared stores. Its value is consumed only after those stores, hiding part of
   the code-to-dictionary latency without changing traffic.
6. The remaining request rounds use the same dense queue.

Packing is safe: there are at most 128 requests, code values fit in 16 bits,
the 2,080-byte warp staging area fits in the 12-bit destination, and a high
half contains one to eight bytes.

Why use a shared queue: compared with the packed-cursor parent, the request
queue improved selected columns by 1.2-7.1%. Nsight Compute showed the reason
was execution shape: eligible warps rose and MIO/short-scoreboard stalls fell,
even though instructions and shared-store conflicts increased. A register-only
queue was 31.7% slower on URL, demonstrating that removing the shared array
also removed useful memory-level parallelism.

Why overlap only the first high load: this preserved the simple bounded queue
while moving one dependent global load ahead of 32 independent low-byte store
instructions in SASS. URL improved another 4.29% with identical global-load
requests, global-load sectors, and global-store sectors. Byte-exact validation
passed on URL, FineWeb text, and HDFS.

### 4. Drain output and advance to the next batch

Both kernels use the same output policy:

1. `chunk_offsets[chunk]` chooses the batch's global output base.
2. A scalar byte head reaches the next 16-byte-aligned output address.
3. Lanes cooperatively drain the aligned body with `uint4` loads and streaming
   `__stcs` stores.
4. A scalar byte tail writes the final partial vector.

There is no write priority or race. Prefix offsets give each token a disjoint
staging interval. After `__syncwarp`, drain iterations give each lane distinct
16-byte body chunks, while head and tail lanes own distinct bytes. Different
warps use disjoint ranges supplied by `chunk_offsets`.

The drain was deliberately left unchanged. A 32-byte-head experiment was
neutral, output-store sectors did not change across the successful cursor and
request-queue experiments, and phase work showed the read/dependency path was
the higher-value target.

## Launch and resource choice

| Property | Baseline | Best kernel | Reason |
| --- | ---: | ---: | --- |
| Tokens per warp | 128 | 128 | Preserve code and output ordering |
| Tokens per lane | 4 | 4 | Preserve coalesced four-plane reads |
| Warps per block | up to 16 | 8 | Allow four 256-thread blocks per SM |
| Launch bound | `__launch_bounds__(512, 2)` | `__launch_bounds__(256, 4)` | Match the measured occupancy target |
| Registers per thread | 64 | 64 | No additional register bottleneck |
| Staging shared memory | 33,280 B max | 16,640 B | Eight warp buffers instead of sixteen |
| Request shared memory | 0 B | 4,096 B | 128 packed requests per warp |
| Total best shared memory | - | 20,736 B | Fits the four-block target on GH200 |
| Spills / stack | none | none | Avoid local-memory traffic |

## Rejected alternatives

- Packing lengths into the code words was rejected because it changes the code
  layout and breaks existing readers.
- Shared-memory lens staging was 36.6% slower on URL because each block copied
  the table and paid reduction/barrier overhead.
- Persistent `cp.async` prefetch was 33.3% slower because duplicated steady
  state/tail control and synchronization outweighed overlap.
- Two packed `scan16x2` scans were 0.96% slower than the single logical-`u64`
  scan and did not reduce the compiled shuffle count.
- Register-only request compaction was 31.7% slower than the shared queue.
- Direct global scatter lost to shared staging and the contiguous aligned
  drain.
- Explicit code `.cg`/`.cs` cache policies did not win universally, so the
  ordinary streaming code load remains.

## Validation status

The best kernel preserved the original `u16` layout and passed poisoned-output
byte comparison on the workloads used for its cursor, request-queue, and
overlap experiments. A fresh direct baseline-versus-best run should be used
for final performance reporting after any build or harness change; older
tables may have different launch partitioning or dispatcher choices.
