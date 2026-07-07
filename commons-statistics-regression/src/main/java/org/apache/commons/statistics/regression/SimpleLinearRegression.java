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
 * Estimates an ordinary least squares regression model with one independent variable:
 *
 * <p>\[ y = \beta_0 + \beta_1 x + \epsilon \]
 *
 * <ul>
 *   <li>Observations \( (x, y) \) can be added to the model one at a time using
 *       {@link #accept(double, double)}, or in bulk using {@link #of(double[], double[])}.</li>
 *   <li>Partial accumulations over separate data may be merged using
 *       {@link #combine(SimpleLinearRegression)}, supporting parallel accumulation.</li>
 *   <li>No data is stored: memory use is constant regardless of the number of
 *       observations.</li>
 *   <li>{@link #getResult()} computes the parameter estimates and summary statistics of
 *       the fit from the current accumulated state; the accumulator may continue to be
 *       updated afterwards.</li>
 * </ul>
 *
 * <p>The parameter estimates require at least two observations with distinct {@code x}
 * values; the standard errors and the error variance additionally require a third
 * observation. Statistics that cannot be computed from the available observations are
 * {@code NaN}.
 *
 * <p>The updating algorithm uses stable mean and co-moment updates as described in
 * Chan, Golub and LeVeque (1983).
 *
 * <p>This class is designed to work with (though does not require)
 * {@linkplain java.util.stream streams}.
 *
 * <p><strong>Note that this implementation is not synchronized.</strong> If
 * multiple threads access an instance of this class concurrently, and at least
 * one of the threads invokes the {@link #accept(double, double) accept} or
 * {@link #combine(SimpleLinearRegression) combine} method, it must be synchronized
 * externally.
 *
 * <p>References:
 * <ul>
 *   <li>Chan, T.F., Golub, G.H. and LeVeque, R.J. (1983)
 *       Algorithms for Computing the Sample Variance: Analysis and Recommendations.
 *       The American Statistician, 37, 242-247.
 *       <a href="https://doi.org/10.2307/2683386">doi: 10.2307/2683386</a></li>
 * </ul>
 *
 * @see <a href="https://en.wikipedia.org/wiki/Simple_linear_regression">
 * Simple linear regression (Wikipedia)</a>
 * @since 1.4
 */
public final class SimpleLinearRegression {
    /** Number of observations. */
    private long n;
    /** Mean of the x values. */
    private double meanX;
    /** Mean of the y values. */
    private double meanY;
    /** Sum of squared deviations of x from its mean. */
    private double sxx;
    /** Sum of squared deviations of y from its mean. */
    private double syy;
    /** Sum of products of the deviations of x and y from their means. */
    private double sxy;

    /** Create an instance. */
    private SimpleLinearRegression() {
        // Do nothing
    }

    /**
     * Creates an instance with no observations.
     *
     * @return {@code SimpleLinearRegression} instance
     */
    public static SimpleLinearRegression create() {
        return new SimpleLinearRegression();
    }

    /**
     * Returns an instance populated using the input observations
     * \( (x_i, y_i) \).
     *
     * @param x Values of the independent variable.
     * @param y Values of the dependent variable.
     * @return {@code SimpleLinearRegression} instance
     * @throws IllegalArgumentException if the arrays differ in length
     */
    public static SimpleLinearRegression of(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                "Values mismatch: " + x.length + " != " + y.length);
        }
        final SimpleLinearRegression r = new SimpleLinearRegression();
        for (int i = 0; i < x.length; i++) {
            r.accept(x[i], y[i]);
        }
        return r;
    }

    /**
     * Updates the state of the model with a new observation.
     *
     * @param x Value of the independent variable.
     * @param y Value of the dependent variable.
     */
    public void accept(double x, double y) {
        final long np1 = n + 1;
        final double dx = x - meanX;
        final double dy = y - meanY;
        final double f = (double) n / np1;
        sxx += dx * dx * f;
        syy += dy * dy * f;
        sxy += dx * dy * f;
        meanX += dx / np1;
        meanY += dy / np1;
        n = np1;
    }

    /**
     * Merges the state of the {@code other} model into this model.
     *
     * @param other Model to be merged.
     * @return {@code this} instance after merging {@code other}
     */
    public SimpleLinearRegression combine(SimpleLinearRegression other) {
        final long m = other.n;
        if (m == 0) {
            return this;
        }
        if (n == 0) {
            n = other.n;
            meanX = other.meanX;
            meanY = other.meanY;
            sxx = other.sxx;
            syy = other.syy;
            sxy = other.sxy;
            return this;
        }
        final long total = n + m;
        final double dx = other.meanX - meanX;
        final double dy = other.meanY - meanY;
        final double f = (double) n * m / total;
        sxx += other.sxx + dx * dx * f;
        syy += other.syy + dy * dy * f;
        sxy += other.sxy + dx * dy * f;
        meanX += dx * m / total;
        meanY += dy * m / total;
        n = total;
        return this;
    }

    /**
     * Gets the number of observations that have been added to the model.
     *
     * @return the number of observations
     */
    public long getN() {
        return n;
    }

    /**
     * Computes the regression parameter estimates and the summary statistics of the fit
     * from the current state of the model.
     *
     * <p>The result contains the intercept \( \beta_0 \) followed by the slope
     * \( \beta_1 \) (see {@link RegressionResult#getCoefficients()}).
     *
     * @return the regression result
     */
    public RegressionResult getResult() {
        final double slope = sxy / sxx;
        final double intercept = meanY - slope * meanX;
        // Sum of squared errors, guarding against negative values from cancellation
        final double sse = Math.max(0, syy - sxy * sxy / sxx);
        final double errorVariance = sse / (n - 2);
        final double slopeError = Math.sqrt(errorVariance / sxx);
        final double interceptError =
            Math.sqrt(errorVariance * (1.0 / n + meanX * meanX / sxx));
        return new Result(n,
            new double[] {intercept, slope},
            new double[] {interceptError, slopeError},
            sse, syy);
    }

    /**
     * Result of a {@link SimpleLinearRegression} fit.
     */
    private static final class Result extends BaseRegressionResult {
        /**
         * Create an instance.
         *
         * @param n Number of observations.
         * @param coefficients Parameter estimates.
         * @param standardErrors Standard errors of the parameter estimates.
         * @param sse Sum of squared errors.
         * @param sst Total sum of squares about the mean.
         */
        Result(long n, double[] coefficients, double[] standardErrors,
               double sse, double sst) {
            super(n, true, coefficients, standardErrors, sse, sst);
        }
    }
}
