/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.statistics.regression;

/**
 * QR decomposition of a rectangular matrix using Householder reflectors, tailored to
 * solving least squares problems.
 *
 * <p>Decomposes an \( n \times p \) matrix \( A \) (with \( n \ge p \)) such that
 * \( A = QR \) where \( Q \) is orthogonal and \( R \) is upper triangular. The matrix
 * is stored in the compact form used by LAPACK: the upper triangle of the decomposed
 * array holds \( R \) (excluding its diagonal, held separately) and the lower part
 * holds the Householder reflector vectors.
 *
 * <p>This class is not part of the public API.
 *
 * @since 1.4
 */
final class QRDecomposition {
    /**
     * Relative threshold used to detect rank deficiency: a column is declared linearly
     * dependent if the norm remaining after applying the previous reflectors is below
     * this fraction of the original column norm.
     */
    private static final double SINGULARITY_THRESHOLD = 1e-12;

    /** Compact representation of the decomposition (reflectors and upper triangle). */
    private final double[][] qr;
    /** Diagonal elements of R. */
    private final double[] rDiag;

    /**
     * Decompose the matrix. The input array is modified in place.
     *
     * @param a Matrix in row-major form, with at least as many rows as columns.
     * @throws IllegalArgumentException if the matrix is rank deficient
     */
    QRDecomposition(double[][] a) {
        final int n = a.length;
        final int p = a[0].length;
        qr = a;
        rDiag = new double[p];
        final double[] colNorm = columnNorms(a);
        for (int k = 0; k < p; k++) {
            // Norm of the lower part of column k
            double normSqr = 0;
            for (int i = k; i < n; i++) {
                normSqr += qr[i][k] * qr[i][k];
            }
            final double alpha = qr[k][k] > 0 ? -Math.sqrt(normSqr) : Math.sqrt(normSqr);
            rDiag[k] = alpha;
            if (Math.abs(alpha) <= SINGULARITY_THRESHOLD * colNorm[k]) {
                throw new IllegalArgumentException(
                    "Rank deficient design matrix at column " + k);
            }
            // Form the reflector v = x - alpha * e_k in place and apply it to the
            // remaining columns
            qr[k][k] -= alpha;
            applyReflector(k, alpha);
        }
    }

    /**
     * Compute the norm of each column of the matrix, used to detect rank deficiency.
     *
     * @param a Matrix.
     * @return the column norms
     */
    private static double[] columnNorms(double[][] a) {
        final double[] norms = new double[a[0].length];
        for (int j = 0; j < norms.length; j++) {
            double s = 0;
            for (final double[] row : a) {
                s += row[j] * row[j];
            }
            norms[j] = Math.sqrt(s);
        }
        return norms;
    }

    /**
     * Apply the Householder reflector held in the lower part of column {@code k} to the
     * columns after {@code k}.
     *
     * @param k Index of the reflector column.
     * @param alpha Signed norm of the reflected column.
     */
    private void applyReflector(int k, double alpha) {
        final int n = qr.length;
        final int p = rDiag.length;
        for (int j = k + 1; j < p; j++) {
            double dot = 0;
            for (int i = k; i < n; i++) {
                dot += qr[i][k] * qr[i][j];
            }
            final double f = dot / (alpha * qr[k][k]);
            for (int i = k; i < n; i++) {
                qr[i][j] += f * qr[i][k];
            }
        }
    }

    /**
     * Solve the least squares problem \( \min \| A \beta - y \| \).
     *
     * @param y Observations (length equal to the row count of the decomposed matrix).
     * @return the solution \( \beta \)
     */
    double[] solve(double[] y) {
        final int n = qr.length;
        final int p = rDiag.length;
        final double[] b = y.clone();
        // Apply the Householder reflectors to y
        for (int k = 0; k < p; k++) {
            double dot = 0;
            for (int i = k; i < n; i++) {
                dot += qr[i][k] * b[i];
            }
            final double f = dot / (rDiag[k] * qr[k][k]);
            for (int i = k; i < n; i++) {
                b[i] += f * qr[i][k];
            }
        }
        // Back-substitute R beta = Q^T y
        final double[] beta = new double[p];
        for (int k = p - 1; k >= 0; k--) {
            double v = b[k];
            for (int j = k + 1; j < p; j++) {
                v -= qr[k][j] * beta[j];
            }
            beta[k] = v / rDiag[k];
        }
        return beta;
    }

    /**
     * Compute \( (A^T A)^{-1} = (R^T R)^{-1} = R^{-1} R^{-T} \),
     * used for the covariance of the least squares parameter estimates.
     *
     * @return the unscaled parameter covariance matrix
     */
    double[][] covariance() {
        final int p = rDiag.length;
        // Invert the upper triangular matrix R by columns: solve R v = e_j
        final double[][] rInv = new double[p][p];
        for (int j = p - 1; j >= 0; j--) {
            rInv[j][j] = 1 / rDiag[j];
            for (int k = j - 1; k >= 0; k--) {
                double v = 0;
                for (int i = k + 1; i <= j; i++) {
                    v -= qr[k][i] * rInv[i][j];
                }
                rInv[k][j] = v / rDiag[k];
            }
        }
        // R^-1 R^-T, exploiting symmetry
        final double[][] cov = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = i; j < p; j++) {
                double sum = 0;
                for (int k = j; k < p; k++) {
                    sum += rInv[i][k] * rInv[j][k];
                }
                cov[i][j] = sum;
                cov[j][i] = sum;
            }
        }
        return cov;
    }

    /**
     * Compute the leverage value \( h_i = a_i^T (A^T A)^{-1} a_i = \| R^{-T} a_i \|^2 \)
     * for each row \( a_i \): the diagonal of the hat matrix.
     *
     * @param rows Rows of the matrix that was decomposed (the decomposition modifies the
     * input in place, so this must be a copy taken before construction).
     * @return the leverage values
     */
    double[] leverage(double[][] rows) {
        final int p = rDiag.length;
        final double[] h = new double[rows.length];
        final double[] z = new double[p];
        for (int i = 0; i < rows.length; i++) {
            final double[] a = rows[i];
            // Forward substitution: R^T z = a
            double sumSq = 0;
            for (int j = 0; j < p; j++) {
                double v = a[j];
                for (int k = 0; k < j; k++) {
                    v -= qr[k][j] * z[k];
                }
                final double zj = v / rDiag[j];
                z[j] = zj;
                sumSq += zj * zj;
            }
            h[i] = sumSq;
        }
        return h;
    }
}
