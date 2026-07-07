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

import java.util.Arrays;

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
        return fitInternal(x, y, null);
    }

    /**
     * Fits the model to the provided data by weighted least squares.
     *
     * <p>The model assumes the variance of the error term for observation {@code i} is
     * inversely proportional to {@code weights[i]}: observations with larger weights
     * carry more information. The parameter estimates minimise the weighted sum of
     * squared residuals \( \sum_i w_i (y_i - \hat{y}_i)^2 \), and the summary statistics
     * (error variance, standard errors, \( R^2 \)) use weighted sums, consistent with
     * the conventions of the R {@code lm} function. {@link Result#getResiduals()}
     * returns the unweighted residuals \( y_i - \hat{y}_i \).
     *
     * <p>See {@link #fit(double[][], double[])} for the data layout and the other
     * conditions on the arguments.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param y Values of the dependent variable.
     * @param weights Weights of the observations.
     * @return the regression result
     * @throws IllegalArgumentException if the arrays are empty or of mismatched lengths,
     * if a weight is not strictly positive and finite, if there are insufficient
     * observations, or if the design matrix is rank deficient
     */
    public Result fit(double[][] x, double[] y, double[] weights) {
        if (weights.length != x.length) {
            throw valuesMismatch(weights.length, x.length);
        }
        for (final double w : weights) {
            if (!(w > 0 && w < Double.POSITIVE_INFINITY)) {
                throw new IllegalArgumentException("Weights must be positive finite: " + w);
            }
        }
        return fitInternal(x, y, weights);
    }

    /**
     * Fits the model to the provided data, optionally by weighted least squares.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param y Values of the dependent variable.
     * @param w Weights of the observations (may be null for an unweighted fit).
     * @return the regression result
     * @throws IllegalArgumentException if the data is invalid
     */
    private Result fitInternal(double[][] x, double[] y, double[] w) {
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
        final double[] weights = w != null ? w : unitWeights(n);
        final double[][] a = designMatrix(x, params, weights);
        // The decomposition modifies the matrix in place; retain the rows for the leverage
        final double[][] rows = new double[n][];
        for (int i = 0; i < n; i++) {
            rows[i] = a[i].clone();
        }
        final QRDecomposition qr = new QRDecomposition(a);
        final double[] beta = qr.solve(weightedResponse(y, weights));

        final double[] residuals = residuals(x, y, beta);
        double sse = 0;
        for (int i = 0; i < n; i++) {
            final double r = residuals[i];
            sse += weights[i] * r * r;
        }
        final double sst = totalSumOfSquares(y, weights);

        // Covariance of the estimates: sigma^2 (X^T W X)^-1
        final double errorVariance = sse / (n - params);
        final double[][] covariance = qr.covariance();
        final double[] errors = new double[params];
        for (int i = 0; i < params; i++) {
            for (int j = 0; j < params; j++) {
                covariance[i][j] *= errorVariance;
            }
            errors[i] = Math.sqrt(covariance[i][i]);
        }

        return new Result(new BaseRegressionResult(n, intercept, beta, errors, sse, sst),
            residuals, qr.leverage(rows), covariance);
    }

    /**
     * Build the design matrix, prepending a constant column when the model includes an
     * intercept. For a weighted fit each row is scaled by the square root of its weight.
     *
     * @param x Values of the regressor variables, one row per observation.
     * @param params Number of model parameters (columns of the design matrix).
     * @param weights Weights of the observations.
     * @return the design matrix
     * @throws IllegalArgumentException if the rows of {@code x} differ in length
     */
    private double[][] designMatrix(double[][] x, int params, double[] weights) {
        final int n = x.length;
        final int k = x[0].length;
        final int offset = params - k;
        final double[][] a = new double[n][params];
        for (int i = 0; i < n; i++) {
            final double[] row = x[i];
            if (row.length != k) {
                throw valuesMismatch(row.length, k);
            }
            final double s = Math.sqrt(weights[i]);
            if (intercept) {
                a[i][0] = s;
            }
            for (int j = 0; j < k; j++) {
                a[i][offset + j] = s * row[j];
            }
        }
        return a;
    }

    /**
     * Create an array of unit weights.
     *
     * @param n Length of the array.
     * @return the weights
     */
    private static double[] unitWeights(int n) {
        final double[] w = new double[n];
        Arrays.fill(w, 1);
        return w;
    }

    /**
     * Scale the response by the square root of the weights, matching the scaling of the
     * design matrix.
     *
     * @param y Values of the dependent variable.
     * @param weights Weights of the observations.
     * @return the scaled response
     */
    private static double[] weightedResponse(double[] y, double[] weights) {
        final double[] wy = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            wy[i] = Math.sqrt(weights[i]) * y[i];
        }
        return wy;
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
     * Compute the (weighted) total sum of squares of the response: centered about the
     * (weighted) mean when the model includes an intercept, otherwise uncentered.
     *
     * @param y Values of the dependent variable.
     * @param weights Weights of the observations.
     * @return the total sum of squares
     */
    private double totalSumOfSquares(double[] y, double[] weights) {
        double sst = 0;
        if (intercept) {
            double sumW = 0;
            double meanY = 0;
            for (int i = 0; i < y.length; i++) {
                sumW += weights[i];
                meanY += weights[i] * (y[i] - meanY) / sumW;
            }
            for (int i = 0; i < y.length; i++) {
                final double d = y[i] - meanY;
                sst += weights[i] * d * d;
            }
        } else {
            for (int i = 0; i < y.length; i++) {
                sst += weights[i] * y[i] * y[i];
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
        /** Leverage values of the observations. */
        private final double[] leverage;
        /** Covariance matrix of the parameter estimates. */
        private final double[][] covariance;

        /**
         * Create an instance. Array arguments are used directly and must not be modified
         * by the caller after construction.
         *
         * @param base Base result.
         * @param residuals Residuals of the fit.
         * @param leverage Leverage values of the observations.
         * @param covariance Covariance matrix of the parameter estimates.
         */
        Result(BaseRegressionResult base, double[] residuals, double[] leverage,
               double[][] covariance) {
            super(base);
            this.residuals = residuals;
            this.leverage = leverage;
            this.covariance = covariance;
        }

        /**
         * Gets the residuals of the fit: the observed response minus the model
         * prediction, \( y_i - \hat{y}_i \), for each observation.
         *
         * <p>For a weighted fit these are the unweighted (raw) residuals.
         *
         * @return the residuals
         */
        public double[] getResiduals() {
            return residuals.clone();
        }

        /**
         * Gets the leverage value of each observation: the diagonal of the hat matrix.
         * A leverage value measures the influence of an observation on its own fitted
         * value; the values sum to the number of model parameters.
         *
         * @return the leverage values
         */
        public double[] getLeverage() {
            return leverage.clone();
        }

        /**
         * Gets the covariance matrix of the parameter estimates,
         * \( \sigma^2 (X^T W X)^{-1} \), where \( W \) is the identity for an unweighted
         * fit. Indices correspond to {@link #getCoefficients()}; the diagonal holds the
         * squares of the standard errors.
         *
         * @return the covariance matrix of the parameter estimates
         */
        public double[][] getCoefficientCovariance() {
            final double[][] c = new double[covariance.length][];
            for (int i = 0; i < c.length; i++) {
                c[i] = covariance[i].clone();
            }
            return c;
        }
    }
}
