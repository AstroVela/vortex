// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.io.Serializable;

/**
 * A pushed-down Bernoulli table sample: each row is kept when its deterministic pseudo-random draw falls inside
 * {@code [lowerBound, upperBound)}. The per-file random sequence is derived from {@code seed} and the file path, so a
 * sampled scan is repeatable for a fixed seed and file set.
 *
 * @param lowerBound inclusive lower bound of the acceptance interval, in {@code [0, 1]}
 * @param upperBound exclusive upper bound of the acceptance interval, in {@code [0, 1]}
 * @param seed the user-supplied sampling seed
 */
public record VortexTableSample(double lowerBound, double upperBound, long seed) implements Serializable {}
