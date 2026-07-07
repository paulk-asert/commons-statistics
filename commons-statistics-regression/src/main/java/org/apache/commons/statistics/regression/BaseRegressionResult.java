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

import org.apache.commons.statistics.distribution.TDistribution;

/**
 * Base implementation of a {@link RegressionResult}. Derives the summary statistics of
 * the fit from the error and total sum of squares.
 *
 * <p>Instances are immutable.
 *
 * @since 1.4
 */
class BaseRegressionResult implements RegressionResult {
    /** Number of observations. */
    private final long n;
    /** Flag indicating the model was fitted with an intercept. */
    private final boolean intercept;
    /** Parameter estimates (intercept first, when fitted). */
    private final double[] coefficients;
    /** Standard errors of the parameter estimates. */
    private final double[] standardErrors;
    /** R-squared statistic. */
    private final double rSquared;
    /** Adjusted R-squared statistic. */
    private final double adjustedRSquared;
    /** Estimated variance of the error term. */
    private final double errorVariance;

    /**
     * Create an instance. Array arguments are used directly and must not be modified
     * by the caller after construction.
     *
     * @param n Number of observations.
     * @param intercept Flag indicating the model was fitted with an intercept.
     * @param coefficients Parameter estimates.
     * @param standardErrors Standard errors of the parameter estimates.
     * @param sse Sum of squared errors.
     * @param sst Total sum of squares (centered when the model has an intercept).
     */
    BaseRegressionResult(long n, boolean intercept,
                         double[] coefficients, double[] standardErrors,
                         double sse, double sst) {
        this.n = n;
        this.intercept = intercept;
        this.coefficients = coefficients;
        this.standardErrors = standardErrors;
        final long df = n - coefficients.length;
        errorVariance = sse / df;
        rSquared = 1 - sse / sst;
        adjustedRSquared = intercept ?
            1 - (1 - rSquared) * (n - 1.0) / df :
            1 - (1 - rSquared) * ((double) n / df);
    }

    /**
     * Create an instance with the state of the {@code source} result.
     *
     * @param source Source to copy.
     */
    BaseRegressionResult(BaseRegressionResult source) {
        n = source.n;
        intercept = source.intercept;
        coefficients = source.coefficients;
        standardErrors = source.standardErrors;
        rSquared = source.rSquared;
        adjustedRSquared = source.adjustedRSquared;
        errorVariance = source.errorVariance;
    }

    /** {@inheritDoc} */
    @Override
    public long getN() {
        return n;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasIntercept() {
        return intercept;
    }

    /** {@inheritDoc} */
    @Override
    public double[] getCoefficients() {
        return coefficients.clone();
    }

    /** {@inheritDoc} */
    @Override
    public double[] getCoefficientStandardErrors() {
        return standardErrors.clone();
    }

    /** {@inheritDoc} */
    @Override
    public double getRSquared() {
        return rSquared;
    }

    /** {@inheritDoc} */
    @Override
    public double getAdjustedRSquared() {
        return adjustedRSquared;
    }

    /** {@inheritDoc} */
    @Override
    public double getErrorVariance() {
        return errorVariance;
    }

    /** {@inheritDoc} */
    @Override
    public double getRegressionStandardError() {
        return Math.sqrt(errorVariance);
    }

    /** {@inheritDoc} */
    @Override
    public double[] getCoefficientPValues() {
        final TDistribution dist = TDistribution.of(degreesOfFreedom());
        final double[] p = new double[coefficients.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = 2 * dist.survivalProbability(Math.abs(coefficients[i] / standardErrors[i]));
        }
        return p;
    }

    /** {@inheritDoc} */
    @Override
    public double[][] getCoefficientConfidenceIntervals(double alpha) {
        if (!(alpha > 0 && alpha < 1)) {
            throw new IllegalArgumentException("Significance level not in (0, 1): " + alpha);
        }
        final double t = TDistribution.of(degreesOfFreedom()).inverseSurvivalProbability(alpha / 2);
        final double[][] ci = new double[coefficients.length][2];
        for (int i = 0; i < ci.length; i++) {
            final double halfWidth = t * standardErrors[i];
            ci[i][0] = coefficients[i] - halfWidth;
            ci[i][1] = coefficients[i] + halfWidth;
        }
        return ci;
    }

    /**
     * Gets the residual degrees of freedom: the number of observations minus the number
     * of estimated parameters.
     *
     * @return the degrees of freedom
     */
    private double degreesOfFreedom() {
        return (double) n - coefficients.length;
    }

    /** {@inheritDoc} */
    @Override
    public double predict(double... x) {
        final int offset = intercept ? 1 : 0;
        final int terms = coefficients.length - offset;
        if (x.length != terms) {
            throw new IllegalArgumentException(
                "Incorrect number of variables: " + x.length + ", expected: " + terms);
        }
        double y = intercept ? coefficients[0] : 0;
        for (int i = 0; i < terms; i++) {
            y += coefficients[i + offset] * x[i];
        }
        return y;
    }
}
