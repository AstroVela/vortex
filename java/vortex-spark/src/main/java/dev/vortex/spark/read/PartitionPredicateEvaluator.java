// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.Map;
import java.util.Set;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Literal;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;

/**
 * Evaluates Spark V2 {@link Predicate}s against the Hive-style partition values parsed from a file path, for
 * partition-level file pruning and selection.
 *
 * <p>Evaluation is three-valued and conservative: a file only counts as a definitive match or non-match when the
 * predicate can be fully decided from its partition values. Predicates (or subtrees) that reference non-partition
 * columns, unsupported operators, or values that cannot be interpreted evaluate to "unknown". Null partition values
 * ({@code __HIVE_DEFAULT_PARTITION__}) follow SQL semantics: comparisons yield unknown, {@code IS_NULL} is true.
 */
public final class PartitionPredicateEvaluator {
    private static final String HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__";

    private PartitionPredicateEvaluator() {}

    /** Returns {@code false} only when at least one predicate definitively evaluates to false. */
    public static boolean matches(Map<String, String> partitionValues, Predicate[] predicates) {
        for (Predicate predicate : predicates) {
            if (evaluate(partitionValues, predicate) == Result.FALSE) {
                return false;
            }
        }
        return true;
    }

    /** Returns {@code true} only when every predicate definitively evaluates to true. */
    public static boolean definitelyMatches(Map<String, String> partitionValues, Predicate[] predicates) {
        for (Predicate predicate : predicates) {
            if (evaluate(partitionValues, predicate) != Result.TRUE) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns whether a predicate can be decided from partition values alone: a boolean combination of {@code =},
     * {@code <=>}, {@code IN}, {@code IS_NULL} and {@code IS_NOT_NULL} over single-name references to the given
     * partition columns (plus {@code ALWAYS_TRUE}/{@code ALWAYS_FALSE}).
     */
    public static boolean isPartitionPredicate(Predicate predicate, Set<String> partitionColumnNames) {
        if (predicate instanceof And and) {
            return isPartitionPredicate(and.left(), partitionColumnNames)
                    && isPartitionPredicate(and.right(), partitionColumnNames);
        }
        if (predicate instanceof Or or) {
            return isPartitionPredicate(or.left(), partitionColumnNames)
                    && isPartitionPredicate(or.right(), partitionColumnNames);
        }
        if (predicate instanceof Not not) {
            return isPartitionPredicate(not.child(), partitionColumnNames);
        }
        Expression[] children = predicate.children();
        switch (predicate.name()) {
            case "ALWAYS_TRUE":
            case "ALWAYS_FALSE":
                return true;
            case "IS_NULL":
            case "IS_NOT_NULL":
                return children.length == 1 && isPartitionColumnRef(children[0], partitionColumnNames);
            case "=":
            case "<=>": {
                if (children.length != 2) {
                    return false;
                }
                if (isPartitionColumnRef(children[0], partitionColumnNames)) {
                    return children[1] instanceof Literal;
                }
                return children[0] instanceof Literal && isPartitionColumnRef(children[1], partitionColumnNames);
            }
            case "IN": {
                if (children.length < 2 || !isPartitionColumnRef(children[0], partitionColumnNames)) {
                    return false;
                }
                for (int i = 1; i < children.length; i++) {
                    if (!(children[i] instanceof Literal)) {
                        return false;
                    }
                }
                return true;
            }
            default:
                return false;
        }
    }

    private static boolean isPartitionColumnRef(Expression expression, Set<String> partitionColumnNames) {
        return expression instanceof NamedReference ref
                && ref.fieldNames().length == 1
                && partitionColumnNames.contains(ref.fieldNames()[0]);
    }

    private enum Result {
        TRUE,
        FALSE,
        UNKNOWN;

        Result negate() {
            return switch (this) {
                case TRUE -> FALSE;
                case FALSE -> TRUE;
                case UNKNOWN -> UNKNOWN;
            };
        }
    }

    private static Result evaluate(Map<String, String> partitionValues, Predicate predicate) {
        if (predicate instanceof And and) {
            Result left = evaluate(partitionValues, and.left());
            Result right = evaluate(partitionValues, and.right());
            if (left == Result.FALSE || right == Result.FALSE) {
                return Result.FALSE;
            }
            return left == Result.TRUE && right == Result.TRUE ? Result.TRUE : Result.UNKNOWN;
        }
        if (predicate instanceof Or or) {
            Result left = evaluate(partitionValues, or.left());
            Result right = evaluate(partitionValues, or.right());
            if (left == Result.TRUE || right == Result.TRUE) {
                return Result.TRUE;
            }
            return left == Result.FALSE && right == Result.FALSE ? Result.FALSE : Result.UNKNOWN;
        }
        if (predicate instanceof Not not) {
            return evaluate(partitionValues, not.child()).negate();
        }

        return switch (predicate.name()) {
            case "ALWAYS_TRUE" -> Result.TRUE;
            case "ALWAYS_FALSE" -> Result.FALSE;
            case "IS_NULL" -> evaluateNullCheck(partitionValues, predicate, true);
            case "IS_NOT_NULL" -> evaluateNullCheck(partitionValues, predicate, false);
            case "=" -> evaluateEquality(partitionValues, predicate, /* nullSafe= */ false);
            case "<=>" -> evaluateEquality(partitionValues, predicate, /* nullSafe= */ true);
            case "IN" -> evaluateIn(partitionValues, predicate);
            default -> Result.UNKNOWN;
        };
    }

    private static Result evaluateNullCheck(Map<String, String> partitionValues, Predicate predicate, boolean isNull) {
        Expression[] children = predicate.children();
        if (children.length != 1 || !(children[0] instanceof NamedReference ref)) {
            return Result.UNKNOWN;
        }
        String value = lookup(partitionValues, ref);
        if (value == null) {
            return Result.UNKNOWN;
        }
        boolean valueIsNull = HIVE_DEFAULT_PARTITION.equals(value);
        return (valueIsNull == isNull) ? Result.TRUE : Result.FALSE;
    }

    private static Result evaluateEquality(Map<String, String> partitionValues, Predicate predicate, boolean nullSafe) {
        Expression[] children = predicate.children();
        if (children.length != 2) {
            return Result.UNKNOWN;
        }
        NamedReference ref = null;
        Literal<?> literal = null;
        if (children[0] instanceof NamedReference r && children[1] instanceof Literal<?> l) {
            ref = r;
            literal = l;
        } else if (children[1] instanceof NamedReference r && children[0] instanceof Literal<?> l) {
            ref = r;
            literal = l;
        } else {
            return Result.UNKNOWN;
        }
        String value = lookup(partitionValues, ref);
        if (value == null) {
            return Result.UNKNOWN;
        }
        boolean valueIsNull = HIVE_DEFAULT_PARTITION.equals(value);
        boolean literalIsNull = literal.value() == null;
        if (nullSafe) {
            // <=> is total: null <=> null is true, null <=> non-null is false.
            if (valueIsNull || literalIsNull) {
                return fromBoolean(valueIsNull && literalIsNull);
            }
            return equalsLiteral(value, literal);
        }
        if (valueIsNull || literalIsNull) {
            // NULL = anything is unknown in SQL; keeping it unknown makes NOT(...) behave.
            return Result.UNKNOWN;
        }
        return equalsLiteral(value, literal);
    }

    private static Result evaluateIn(Map<String, String> partitionValues, Predicate predicate) {
        Expression[] children = predicate.children();
        if (children.length < 2 || !(children[0] instanceof NamedReference ref)) {
            return Result.UNKNOWN;
        }
        String value = lookup(partitionValues, ref);
        if (value == null) {
            return Result.UNKNOWN;
        }
        if (HIVE_DEFAULT_PARTITION.equals(value)) {
            // NULL IN (...) is unknown in SQL.
            return Result.UNKNOWN;
        }
        boolean sawUnknown = false;
        for (int i = 1; i < children.length; i++) {
            if (!(children[i] instanceof Literal<?> literal)) {
                sawUnknown = true;
                continue;
            }
            switch (equalsLiteral(value, literal)) {
                case TRUE -> {
                    return Result.TRUE;
                }
                case UNKNOWN -> sawUnknown = true;
                case FALSE -> {
                    // keep scanning the IN list
                }
            }
        }
        return sawUnknown ? Result.UNKNOWN : Result.FALSE;
    }

    private static String lookup(Map<String, String> partitionValues, NamedReference ref) {
        if (ref.fieldNames().length != 1) {
            return null;
        }
        return partitionValues.get(ref.fieldNames()[0]);
    }

    /**
     * Compares a partition value string with a literal by interpreting the string in the literal's type using the same
     * internal representations the partitioned writer emits (epoch days for dates, epoch micros for timestamps).
     */
    private static Result equalsLiteral(String value, Literal<?> literal) {
        Object literalValue = literal.value();
        if (literalValue == null) {
            return Result.UNKNOWN;
        }
        DataType type = literal.dataType();
        try {
            if (type instanceof StringType) {
                return fromBoolean(literalValue.toString().equals(value));
            } else if (type instanceof IntegerType || type instanceof DateType) {
                return fromBoolean(Integer.parseInt(value) == ((Number) literalValue).intValue());
            } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
                return fromBoolean(Long.parseLong(value) == ((Number) literalValue).longValue());
            } else if (type instanceof ShortType) {
                return fromBoolean(Short.parseShort(value) == ((Number) literalValue).shortValue());
            } else if (type instanceof ByteType) {
                return fromBoolean(Byte.parseByte(value) == ((Number) literalValue).byteValue());
            } else if (type instanceof BooleanType) {
                return fromBoolean(Boolean.parseBoolean(value) == (Boolean) literalValue);
            } else if (type instanceof DoubleType) {
                return fromBoolean(Double.parseDouble(value) == ((Number) literalValue).doubleValue());
            } else if (type instanceof FloatType) {
                return fromBoolean(Float.parseFloat(value) == ((Number) literalValue).floatValue());
            }
        } catch (NumberFormatException | ClassCastException e) {
            return Result.UNKNOWN;
        }
        return Result.UNKNOWN;
    }

    private static Result fromBoolean(boolean value) {
        return value ? Result.TRUE : Result.FALSE;
    }
}
