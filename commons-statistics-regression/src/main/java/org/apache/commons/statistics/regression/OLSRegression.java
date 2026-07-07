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
 * Estimates an ordinary least squares (OLS) multiple linear regression model:
 *
 * <p>\[ y = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k + \epsilon \]
 *
 * <p>The model is fitted by {@link #fit(double[][], double[])} using a QR decomposition
 * of the design matrix. By default the model includes an intercept term \( \beta_0 \);
 * use {@link #withIntercept(boolean)} to fit a model through the origin.
 *
 * <p>Example:
 *
 * <pre>
 * double[][] x = { {1, 9}, {2, 4}, {3, 7}, {4, 3}, {5, 1} };
 * double[] y = {6, 5, 8, 7, 9};
 * OLSRegression.Result r = OLSRegression.withDefaults().fit(x, y);
 * double[] beta = r.getCoefficients();
 * double rSquared = r.getRSquared();
 * </pre>
 *
 * <p>Instances of this class are immutable.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Ordinary_least_squares">
 * Ordinary least squares (Wikipedia)</a>
 * @since 1.4
 */
public final class OLSRegression {
    /** Default instance. */
    private static final OLSRegression DEFAULT = new OLSRegression(true);

    /** Flag indicating the model includes an intercept term. */
    private final boolean intercept;

    /**
     * Create an instance.
     *
     * @param intercept Flag indicating the model includes an intercept term.
     */
    private OLSRegression(boolean intercept) {
        this.intercept = intercept;
    }

    /**
     * Return an instance using the default options.
     *
     * <ul>
     *   <li>{@code intercept = true}</li>
     * </ul>
     *
     * @return default instance
     */
    public static OLSRegression withDefaults() {
        return DEFAULT;
    }

    /**
     * Return an instance with the configured intercept option.
     *
     * <p>If {@code true} an intercept term \( \beta_0 \) is estimated; the design matrix
     * passed to {@link #fit(double[][], double[])} must not include a constant column.
     * If {@code false} the model is fitted through the origin.
     *
     * @param v Value.
     * @return an instance
     */
    public OLSRegression withIntercept(boolean v) {
        if (v == intercept) {
            return this;
        }
        return v ? DEFAULT : new OLSRegression(false);
    }

    /**
     * Fits the model to the provided data.
     *
     * <p>Row {@code i} of {@code x} holds the values of the \( k \) regressor variables
     * for observation {@code i}, whose response is {@code y[i]}. If the model includes an
     * intercept (the default) a constant term is added internally and must not be present
     * in {@code x}.
     *
     * <p>The number of observations must exceed the number of estimated parameters.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param y Values of the dependent variable.
     * @return the regression result
     * @throws IllegalArgumentException if the arrays are empty or of mismatched lengths,
     * if there are insufficient observations, or if the design matrix is rank deficient
     * (for example, when two regressor variables are linearly dependent)
     */
    public Result fit(double[][] x, double[] y) {
        final int n = x.length;
        if (n == 0) {
            throw new IllegalArgumentException("No data");
        }
        if (n != y.length) {
            throw valuesMismatch(n, y.length);
        }
        final int k = x[0].length;
        final int params = k + (intercept ? 1 : 0);
        if (n <= params) {
            throw new IllegalArgumentException(
                "Not enough observations: " + n + " <= " + params);
        }
        final QRDecomposition qr = new QRDecomposition(designMatrix(x, params));
        final double[] beta = qr.solve(y);

        final double[] residuals = residuals(x, y, beta);
        double sse = 0;
        for (final double r : residuals) {
            sse += r * r;
        }
        final double sst = totalSumOfSquares(y);

        // Standard errors from the diagonal of sigma^2 (X^T X)^-1
        final double errorVariance = sse / (n - params);
        final double[] cov = qr.covarianceDiagonal();
        final double[] errors = new double[params];
        for (int i = 0; i < params; i++) {
            errors[i] = Math.sqrt(errorVariance * cov[i]);
        }

        return new Result(n, intercept, beta, errors, sse, sst, residuals);
    }

    /**
     * Build the design matrix, prepending a constant column when the model includes
     * an intercept.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param params Number of model parameters (columns of the design matrix).
     * @return the design matrix
     * @throws IllegalArgumentException if the rows of {@code x} differ in length
     */
    private double[][] designMatrix(double[][] x, int params) {
        final int n = x.length;
        final int k = x[0].length;
        final int offset = params - k;
        final double[][] a = new double[n][params];
        for (int i = 0; i < n; i++) {
            final double[] row = x[i];
            if (row.length != k) {
                throw valuesMismatch(row.length, k);
            }
            if (intercept) {
                a[i][0] = 1;
            }
            System.arraycopy(row, 0, a[i], offset, k);
        }
        return a;
    }

    /**
     * Compute the residuals of the fit: the observed response minus the model prediction.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param y Values of the dependent variable.
     * @param beta Parameter estimates.
     * @return the residuals
     */
    private double[] residuals(double[][] x, double[] y, double[] beta) {
        final int offset = intercept ? 1 : 0;
        final double[] residuals = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            double yHat = intercept ? beta[0] : 0;
            final double[] row = x[i];
            for (int j = 0; j < row.length; j++) {
                yHat += beta[j + offset] * row[j];
            }
            residuals[i] = y[i] - yHat;
        }
        return residuals;
    }

    /**
     * Compute the total sum of squares of the response: centered about the mean when the
     * model includes an intercept, otherwise uncentered.
     *
     * @param y Values of the dependent variable.
     * @return the total sum of squares
     */
    private double totalSumOfSquares(double[] y) {
        double sst = 0;
        if (intercept) {
            double meanY = 0;
            for (int i = 0; i < y.length; i++) {
                meanY += (y[i] - meanY) / (i + 1);
            }
            for (final double v : y) {
                final double d = v - meanY;
                sst += d * d;
            }
        } else {
            for (final double v : y) {
                sst += v * v;
            }
        }
        return sst;
    }

    /**
     * Create an exception for mismatched array lengths.
     *
     * @param actual Actual length.
     * @param expected Expected length.
     * @return the exception
     */
    private static IllegalArgumentException valuesMismatch(int actual, int expected) {
        return new IllegalArgumentException(
            "Values mismatch: " + actual + " != " + expected);
    }

    /**
     * Result of an {@link OLSRegression} fit.
     *
     * @since 1.4
     */
    public static final class Result extends BaseRegressionResult {
        /** Residuals of the fit. */
        private final double[] residuals;

        /**
         * Create an instance.
         *
         * @param n Number of observations.
         * @param intercept Flag indicating the model was fitted with an intercept.
         * @param coefficients Parameter estimates.
         * @param standardErrors Standard errors of the parameter estimates.
         * @param sse Sum of squared errors.
         * @param sst Total sum of squares (centered when the model has an intercept).
         * @param residuals Residuals of the fit.
         */
        Result(long n, boolean intercept, double[] coefficients, double[] standardErrors,
               double sse, double sst, double[] residuals) {
            super(n, intercept, coefficients, standardErrors, sse, sst);
            this.residuals = residuals;
        }

        /**
         * Gets the residuals of the fit: the observed response minus the model
         * prediction, \( y_i - \hat{y}_i \), for each observation.
         *
         * @return the residuals
         */
        public double[] getResiduals() {
            return residuals.clone();
        }
    }
}
