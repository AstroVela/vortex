# Spark

Vortex provides a Spark DataSource V2 connector for reading and writing Vortex files. The
connector is published to Maven Central as `dev.vortex:vortex-spark_2.13` (the artifact name carries
the Scala binary version; `vortex-spark_2.12` is also published).

## Installation

Add the dependency to your build. The `vortex-spark_2.13` artifact targets Spark 4.x; a
`vortex-spark_2.12` artifact targeting Spark 3.x is also published.

````{tab} Gradle (Kotlin)
```kotlin
implementation("dev.vortex:vortex-spark_2.13:<version>")
```
````

````{tab} Maven
```xml
<dependency>
    <groupId>dev.vortex</groupId>
    <artifactId>vortex-spark_2.13</artifactId>
    <version>${vortex.version}</version>
</dependency>
```
````

The connector ships as a shadow JAR that relocates its Arrow, Guava, and Protobuf dependencies
to avoid classpath conflicts with Spark.

## Reading Vortex Files

Use the `vortex` format to read a single file or a directory of Vortex files:

```java
Dataset<Row> df = spark.read()
    .format("vortex")
    .option("path", "/path/to/data.vortex")
    .load();
```

When pointed at a directory, the connector discovers all `.vortex` files and creates one read
partition per file.

Column pruning is pushed down — only the columns referenced by the query are read from the file.

## Writing Vortex Files

```java
df.write()
    .format("vortex")
    .option("path", "/path/to/output")
    .mode(SaveMode.Overwrite)
    .save();
```

Each write task writes one `part-%05d-%d.vortex` file per partition path it touches (just one file
for an unpartitioned write); the zero-padded partition id and the task id (e.g. `part-00000-3.vortex`)
keep the name unique within each path.

### Write Options

| Option                    | Default | Description                        |
|---------------------------|---------|------------------------------------|
| `vortex.write.batch.size` | 2048    | Number of rows per batch (1–65536) |

### Save Modes

The connector supports all standard Spark save modes: `Overwrite`, `Append`, `Ignore`, and
`ErrorIfExists`.

## Supported Types

| Spark Type         | Vortex Type                            |
|--------------------|----------------------------------------|
| `BooleanType`      | Bool                                   |
| `ByteType`         | Int8 / UInt8                           |
| `ShortType`        | Int16 / UInt16                         |
| `IntegerType`      | Int32 / UInt32                         |
| `LongType`         | Int64 / UInt64                         |
| `FloatType`        | Float32                                |
| `DoubleType`       | Float64                                |
| `StringType`       | Utf8                                   |
| `BinaryType`       | Binary                                 |
| `DecimalType`      | Decimal                                |
| `DateType`         | Date (days)                            |
| `TimestampType`    | Timestamp (microseconds, UTC)          |
| `TimestampNTZType` | Timestamp (microseconds, no timezone)  |
| `ArrayType`        | List                                   |
| `StructType`       | Struct                                 |

## S3 Support

The connector supports reading and writing to S3 paths:

```java
Dataset<Row> df = spark.read()
    .format("vortex")
    .option("path", "s3://bucket/path/to/data")
    .load();
```
