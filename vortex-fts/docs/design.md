# Vortex Tantivy 全文检索设计

## 目标与边界

为仓库的 Vortex 文件扫描能力增加可持久化全文检索，复用 Tantivy 的分词、倒排表、
查询解析和 BM25。实现位于 `vortex-fts` workspace crate；DuckDB 扩展按 feature 注册后端，
已有 Vortex 读写流程和文件格式不变。

本版本包含单文件 CLI/Rust API、`vortex-index` 多文件全文后端、DuckDB SQL 以及配套 Iceberg 显式快照适配器。未实现 SQL 优化器自动改写、云端索引存储、增量事务维护、中文专用分词和高亮。下面先描述保留的单文件 API，再描述通用后端与集成。

## 数据布局与模块

```text
articles.vortex                 # 原始列式数据，不修改
articles.fts/
  manifest.json                 # 完成标记、版本、源身份和列配置
  index/                        # Tantivy 持久化索引
    meta.json
    ... segment 文件 ...
```

| 模块 | 职责 |
| --- | --- |
| `src/lib.rs` | 有序扫描建索引、查询、结果映射、回表 |
| `src/schema.rs` | 字段校验、Tantivy schema 构造 |
| `src/manifest.rs` | 版本化元数据、SHA-256 身份、清单发布 |
| `src/bin/vortex-fts.rs` | build/search/inspect CLI 与 JSON 输出 |
| `src/tests.rs` | 查询、行号、NULL、重启、故障与格式兼容测试 |

清单版本为 1，记录源字节数、SHA-256、物理行数、索引列的有序列表和
`tantivy-0.25-default` analyzer 标识。Tantivy 依赖固定为 0.25.0；索引编码由 Tantivy
自身校验，应用同时验证清单版本、analyzer、schema 和文档数。版本升级需要评估
兼容性，不能仅改变清单值；不兼容时重建。

## 索引 schema 和行号

每个 Vortex 物理行生成一个 Tantivy document，包含：

- `_vortex_row_id: u64`，stored，记录零基文件行位置；不作为全文字段。
- 每个选定列对应一个 `TEXT` 字段，保存词项、词频、位置；原始字符串不 stored。

一个物理行只生成一个 document，因此跨列命中不会重复返回该行。NULL、空字符串和
顶层 NULL struct 仍生成只有行号的 document；它们计入索引文档数和 `*` 查询。
Tantivy 的 segment 合并会重排内部 DocAddress，但 stored row_id 保持不变。

列名限定为 ASCII 标识符，拒绝保留字段名、重复选择和源 schema 中同名歧义列。
原始行号不等于业务 ID，应用不能在源文件追加/排序后继续使用旧映射。

## 构建流程

1. 流式计算源内容 SHA-256，打开 Vortex 文件并验证所选字段。
2. 通过 `create_dir` 独占创建目标目录；已存在则失败。创建内部临时索引目录。
3. 建立 Tantivy schema 和单线程 writer，设置 64 MB 写预算。
4. 只投影选定列，显式启用有序扫描并设扫描并发为 1，每批至多 8192 行，累加物理行号。
5. 每批 canonicalize struct 和 UTF-8 列，批量物化顶层/列级 validity。
   行循环只访问已解码字符串和 mask，不逐行调用压缩数组 scalar accessor。
6. 检查扫描行数等于 footer 行数，commit 并等待 merge 线程完成。
7. 再次计算源内容身份，确认构建期间的最终内容与初始内容一致。
8. 将完成的临时索引重命名为 `index/`，最后用临时文件原子发布 `manifest.json`。

读者必须先成功读取清单。发布前的索引不可打开；中断可能留下无清单的目录，恢复
策略是清理后重建或使用新目录。清单写入先 sync 再以不覆盖方式发布。该协议避免
正常进程中暴露半成品，但不声称提供断电后跨目录的完整持久化事务。

源文件必须在整个操作期间不可变。前后摘要可以识别普通修改，不能防止恶意或偶然
发生的“修改后恢复”并发竞态；本实现没有锁定源文件或创建快照。应用应通过不可变
对象发布、外部锁或文件生命周期约束保证这一前提。

## 查询与回表

`SearchIndex::open` 打开 index reader 并验证清单/schema/文档数；对象可复用。
`search` 先校验源摘要，再以所有文本字段为默认字段严格解析 Tantivy 查询。
正常查询在同一次搜索中组合 Count 与 TopDocs；limit=0 只执行 Count，避免
Tantivy 的非零 top-k 前提。limit 上限 10000 控制结果分配，空查询显式报错。

Tantivy BM25 负责排序，分数并列顺序未定义。逐个读取少量 stored document，提取
并检查 row_id 范围，返回 `total_hits` 与 `SearchHit { row_id, score }`。

回表接口接收严格递增的行号，通过 `StrictSortedBuffer` 和 Vortex
`with_row_indices` 选择读取原始列。回表顺序是文件顺序；调用者保留 relevance 列表
并在需要时关联重排。回表前后校验源摘要。空结果保留原始 schema，重复/乱序/越界
行号报错，防止将不满足扫描前提的数据传给底层。

## 性能与取舍

旁路索引避免扩展 Vortex footer 和 layout，Tantivy 使用原生文件目录/mmap，代价是
应用需要同时管理数据文件和索引。新增依赖只出现在新 crate 中，不向现有核心 crate
引入 Tantivy。构建采用列投影和批量解码，不完整物化源文件。

内容哈希用于正确识别同大小替换或相同 schema 的错误文件；仅验证路径、mtime 或
行数不足以保证正确性。当前每次查询都读全文件计算摘要，牺牲大文件检索性能。
未来可增加由外部存储层认证的不可变版本 ID，或持有已验证快照的 reader，避免重复
哈希；在提供严格身份保证前不应默认跳过验证。

TopDocs 的内存主要与 limit/segment 数相关，Count 需要遍历全部匹配。写入预算不
包含解码批次、查询结果及 merge 开销。没有将本实现的性能与扫描作未经测量的比较；
实际部署需按文本长度、词项分布、选择率、文件大小和缓存状态测量端到端延迟。

## 验证策略

自动测试覆盖词项、短语、AND/NOT、字段限定、大小写归一化、无命中、空表、空查询、
非法语法、top-k 与总数、count-only、重开、内容不变迁移、同大小源变更、禁止覆盖、
不完整索引、清单版本/tokenizer/schema/行数不一致、多批次行号、NULL struct、NULL
文本、空字符串、选择性回表及非法行号。测试通过正常 Vortex writer 生成文件，覆盖
默认压缩写入和解码路径；可执行 quickstart 验证文档示例流程。当前仓库 writer 拒绝顶层 nullable struct，
因此父级 NULL 的屏蔽语义在批次索引边界直接测试，文件级测试覆盖 nullable 文本列。

依赖升级必须重新执行这些测试，尤其关注 tokenizer 行为、Tantivy schema 和文件
格式兼容性。CLI 还应运行 build/search/inspect 的进程级 smoke test，检查 JSON 和
失败退出码。检查结果在对应 PR 中记录。


## 通用索引能力与多文件后端

`vortex-index::Index` 新增默认返回 None 的 `as_full_text()`，兼容已有 vector/scalar 后端；`FullTextIndex::search_text` 返回 `FullTextHit { row: RowAddress, score: f32 }`，分数降序。全文相关性与向量距离分开表达。

`TantivyProvider`（ID `tantivy.static`，版本 1）同时实现 builder/provider。消费 `IndexSource` 的可见行流，验证行地址、覆盖范围、重复地址与上限，把所有覆盖文件写进同一个 Tantivy index。`_vortex_file_id` 和 `_vortex_row_id` 为 fast u64 字段，与原始行映射解耦于 Tantivy segment/doc ID。文档 NULL 规则与单文件实现相同。默认 analyzer 固定到 Tantivy 0.25.0。

新布局为 reference JSON → LocalGeneration → 校验清单与 artifacts/tantivy/{descriptor.json,index/*}。LocalIndexStore 对 artifact 校验大小、SHA-256、路径类型和快照，打开时 materialize 到私有 lease，native handle 释放后才能删除 lease。后端 descriptor 再验证 metadata identity、schema、analyzer、文档数。构建必须 seal 并重开成功，重新验证源文件后才原子发布 reference；失败不会发布可用 reference。

查询创建单个 Searcher 和全局评分 Weight，逐 segment 枚举命中文档，在加入大小为 k 的堆之前应用 RowFilter，因而不会发生“先 top-k 后过滤导致漏掉剩余候选”的错误。堆和最终排序使用 score DESC、RowAddress ASC。时间复杂度 O(命中数 × log k)，结果存储 O(k)；这是精确检索，不采用 block-max WAND 剪枝。对任意 allow/exclude 过滤，BM25 仍使用索引固定语料统计；Iceberg 删除行在建索引前移除，因此不进入该语料。

## DuckDB 生命周期

`vortex_fts_build(files, reference, fields)` 和 `vortex_fts_search(reference, query, k)` 复用已有 index FFI、访问策略、prepared owner、预算缓存和回表路径。build 仅在执行阶段产生副作用，EXPLAIN 不写索引。Reference 保留完整结果 dtype，bind 阶段确定 `row STRUCT` 返回类型，执行时再次验证 reference identity。

缓存现在保存一般 Index，具体查询入口验证 full-text 或 squared-L2 capability。全文查询支持多个字段，向量接口仍要求单字段；两类查询不能混用。strict 模式重新验证源文件；snapshot 模式必须有 prepared owner。SQL 返回 score，不将 relevance 伪装成 distance。回表通过 IndexSource::take 保持排名顺序，并验证返回行映射和 dtype。

## Iceberg 快照适配

配套 Python orchestration 使用调用方连接执行原生 Iceberg scan、Vortex COPY 和全文 SQL。扫描指定 snapshot ID，复用成熟的删除可见性、字段 ID 和 schema 映射。只物化该快照的可见行，避免自行重写 position/equality-delete 规则。全表物化也让所有数据文件共享一致的 BM25 统计。

发布顺序：独占创建新目录 → snapshot scan/COPY（最多读取 1000001 行，用额外一行检测超限）→ native FTS build/seal/reopen → 再核对 metadata 字节 → 原子发布 iceberg-fts.json。异常清理本次独占目录。目录中的完成标记绑定 table UUID、完整 snapshot 描述、schema 与被索引字段 ID、native reference hash。没有完成标记的目录不可查询。

检索要求调用方再次指定 metadata 和 snapshot，身份不一致或快照过期即失败。业务行来自物化快照，source hash 防止物化文件被替换。读取持久快照遵循 Iceberg 的不可变文件约定，不在每次全文查询重新读取所有远端 manifest/data files。返回地址明确属于物化文件，不是 Iceberg 物理地址。该方案增加存储和全量构建开销；增量更新、自动 catalog latest 跟踪、原文件按行回表属于后续架构工作，不由当前 API 暗示支持。
