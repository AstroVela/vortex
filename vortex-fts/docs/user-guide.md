# Vortex Tantivy 全文检索用户文档

## 能力与使用前提

`vortex-fts` 为一个本地、不可变的 Vortex 文件建立独立全文索引目录。支持：

- 对多个顶层 UTF-8 列同时索引，支持可空列和压缩编码。
- 词项、短语、字段限定、布尔组合查询；按 Tantivy BM25 返回 top-k。
- 返回总命中数和零基物理行号，通过 Rust API 选择读取原文件中的命中行。
- 索引落盘后跨进程重新打开；查询前校验源文件内容身份。

需要仓库指定的 Rust 1.91 工具链。本功能是独立 crate 和 CLI，尚未提供
DuckDB/DataFusion SQL 函数、Python API、对象存储索引或多文件全局搜索。

## 编译与完整示例

在仓库根目录执行：

```sh
cargo build --release -p vortex-fts
cargo run -p vortex-fts --example quickstart
```

`quickstart` 会创建临时 Vortex 数据、构建索引、查询短语、回表读取并检查结果，
退出时清理临时目录。以下示例假设已有包含 `title`、`body` 两列的 `articles.vortex`。
也可以使用 `cargo run -p vortex-fts --` 代替下面的可执行文件路径。

## 建立索引

```sh
./target/release/vortex-fts build \
  --source articles.vortex \
  --index articles.fts \
  --columns title,body
```

成功时标准输出为 JSON 元数据。`articles.fts` 必须不存在，其父目录必须存在；
已有文件或目录不会被覆盖。所选列必须存在且名称唯一，类型必须为 UTF-8。
列名限定为 ASCII 标识符 `[A-Za-z_][A-Za-z0-9_]*`，内部名称 `_vortex_row_id` 保留。
不支持嵌套路径、二进制列和数值列全文索引。

每个物理行对应一个 Tantivy 文档。NULL 和空字符串不产生文本词项；顶层 struct
为 NULL 时其子字段均不参与检索。它们仍保留物理行号，也会被 `*` 查询命中。当前仓库 writer 尚不支持写入顶层
nullable struct；本组件的批次索引逻辑会保留该 NULL 语义。

## 查询

```sh
./target/release/vortex-fts search \
  --source articles.vortex --index articles.fts \
  --query 'body:"full text" AND title:search' --limit 10
```

输出示意（分数随数据而变化）：

```json
{
  "total_hits": 23,
  "hits": [
    { "row_id": 42, "score": 2.15 },
    { "row_id": 7, "score": 1.83 }
  ]
}
```

`row_id` 是源文件中的零基物理位置，不是业务主键，也不是 Tantivy 的 segment/doc ID。
`hits` 按相关性降序排列；分数相同时不保证稳定顺序。`total_hits` 是完整查询的命中数，
不受 `--limit` 限制。默认 limit 为 10，范围 0–10000，0 表示只统计总数。

| 查询 | 语义 |
| --- | --- |
| `rust` | 在所有已索引列中匹配词项 |
| `rust vortex` | 默认 OR，任一词项匹配即可 |
| `rust AND vortex` | 两个词项均匹配 |
| `rust NOT python` | 匹配 rust 且排除 python |
| `"full text"` | 同一字段内按相邻位置匹配短语 |
| `title:rust` | 仅搜索 title 字段 |
| `(rust OR vortex) AND search` | 使用括号组合条件 |
| `*` | 匹配全部物理行，包括空文本行 |

使用 shell 单引号包裹查询，保留短语的双引号。空查询、未知字段和非法语法返回错误，
不降级为部分查询。CLI 出错时写入标准错误并以非零状态退出。

### 分词行为

使用 Tantivy 0.25 默认 analyzer：按字母数字边界切词、过滤 UTF-8 长度达到 40 字节的 token、
转小写，并记录词频和位置。索引与查询使用同一 analyzer。默认不做词干化、停用词
删除、拼音或中文词典分词。连续中文文本可能作为一个 token，不能把本功能当作中文
子串检索；中日韩语言专用分词需要后续新增、版本化 tokenizer 后重建索引。

全文检索不是 SQL `LIKE '%text%'`。短语不能跨列匹配，NULL 也不是可搜索字符串。

## Rust API 与回表

完整、可执行代码见 [quickstart.rs](../examples/quickstart.rs)。核心调用如下：

```rust,ignore
let metadata = build_index(
    &session, &source, &index_dir, &["title".into(), "body".into()],
).await?;
let index = SearchIndex::open(&index_dir)?;
let results = index.search(&source, "\"full text\"", 10)?;
let mut row_ids: Vec<u64> = results.hits.iter().map(|hit| hit.row_id).collect();
row_ids.sort_unstable();
let rows = index.read_rows(&session, &source, &row_ids).await?;
```

`session` 应在 Tokio runtime 内通过 `VortexSession::default()` 创建。
`read_rows` 要求严格升序、无重复且不越界的行号，返回完整原始字段，顺序是文件顺序。
如需相关性顺序，保留 `results.hits` 中的顺序，在应用层按 row_id 关联回表结果。
空选择返回保留源 schema 的空数组。不要把 top-k 行号当作完整的过滤结果集。

## 检查、迁移与重建

```sh
./target/release/vortex-fts inspect --index articles.fts
```

`inspect` 校验清单版本、schema 和文档数，显示索引列、行数、tokenizer 与源 SHA-256。
它不接收源文件，因此不会检查源内容；`search` 和 `read_rows` 会进行该校验。

索引绑定文件内容，不绑定绝对路径。源文件按字节复制或移动后，可提供新路径继续搜索。
覆盖、追加、压缩重写或重排行后必须重建，即使行数或文件大小不变也一样。

重建时输出到新目录，例如 `articles.fts.v2`，成功后由应用切换索引路径。当前版本没有
原地更新、增量追加、删除文档或自动重建。使用期间不得修改源文件或索引目录；校验并
不提供文件系统快照、锁或针对并发写入的事务隔离。

## 故障处理与资源开销

| 现象 | 处理 |
| --- | --- |
| 目标目录已存在 | 使用新目录，避免覆盖有效索引 |
| 缺少 `manifest.json` | 构建未完成；确认没有构建进程后清理该不完整目录再重试 |
| 源文件不匹配 | 使用原文件或重建，不能只编辑清单绕过校验 |
| 版本/tokenizer/schema/文档数不匹配 | 使用匹配版本或重建整个索引 |
| 未知字段 | 使用建索引时选定的列名 |
| 非 UTF-8 或重复列名 | 修正数据 schema 或索引列配置 |

索引使用一个 Tantivy 写线程、64 MB 写入预算，每批至多扫描 8192 行，另需解码字符串、异步任务
及 segment merge 的内存和临时磁盘；64 MB 不是进程内存上限。索引只保存行号和倒排
数据，不额外保存原始文本。回表会物化所选行，应用应控制结果数量。

建索引前后各完整读取文件计算 SHA-256；每次 search 完整读取一次，read_rows 前后
各读取一次。因此大文件查询延迟包含 O(文件字节数) 的顺序 I/O，不承诺低延迟。
异步建索引/回表函数内部也含同步哈希与 Tantivy 操作；在线服务应放到专用工作线程
或独立任务进程中。建议先用业务数据测量索引大小、构建吞吐和端到端查询延迟。
