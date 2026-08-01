// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.connector.catalog.MetadataColumn;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

/**
 * Metadata columns exposed by Vortex tables.
 *
 * <p>{@code _file} is the path of the Vortex file a row was read from; {@code _pos} is the row's position within that
 * file, assigned before any pushed filter (filtered-out rows leave gaps, surviving rows keep their original positions).
 * A table whose data schema already contains a column with one of these names shadows the metadata column.
 */
public final class VortexMetadataColumns {

    /** Name of the file-path metadata column. */
    public static final String FILE_PATH = "_file";

    /** Name of the row-position metadata column. */
    public static final String ROW_POSITION = "_pos";

    private static final MetadataColumn[] ALL = {
        new VortexMetadataColumn(FILE_PATH, DataTypes.StringType, "path of the Vortex file the row belongs to"),
        new VortexMetadataColumn(ROW_POSITION, DataTypes.LongType, "row position inside the Vortex file")
    };

    private VortexMetadataColumns() {}

    /** All metadata columns Vortex tables can serve, as a fresh array. */
    public static MetadataColumn[] all() {
        return ALL.clone();
    }

    private record VortexMetadataColumn(String name, DataType dataType, String comment) implements MetadataColumn {
        @Override
        public boolean isNullable() {
            return false;
        }
    }
}
