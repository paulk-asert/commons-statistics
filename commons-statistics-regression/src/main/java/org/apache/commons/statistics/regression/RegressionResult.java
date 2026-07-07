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
 * Represents the result of fitting a linear regression model
 *
 * <p>\[ y = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k + \epsilon \]
 *
 * <p>by ordinary least squares. If the model was fitted without an intercept the
 * \( \beta_0 \) term is omitted.
 *
 * @since 1.4
 */
public interface RegressionResult {
    /**
     * Gets the number of observations used to fit the model.
     *
     * @return the number of observations
     */
    long getN();

    /**
     * Indicates whether the model was fitted with an intercept term.
     *
     * @return true if the model includes an intercept
     */
    boolean hasIntercept();

    /**
     * Gets the estimates of the regression parameters \( \beta \).
     *
     * <p>If the model was fitted with an intercept (see {@link #hasIntercept()}) then the
     * intercept estimate \( \beta_0 \) is the first element; the estimate for regressor
     * variable \( x_j \) is at index {@code j}. Otherwise the estimate for \( x_j \) is
     * at index {@code j - 1}.
     *
     * @return the parameter estimates
     */
    double[] getCoefficients();

    /**
     * Gets the standard errors of the regression parameters. Indices correspond to
     * {@link #getCoefficients()}.
     *
     * @return the standard errors of the parameter estimates
     */
    double[] getCoefficientStandardErrors();

    /**
     * Gets the coefficient of determination \( R^2 \), the proportion of the variation in
     * the response explained by the model.
     *
     * <p>If the model was fitted with an intercept this uses the total sum of squares about
     * the mean of {@code y}; otherwise the uncentered total sum of squares is used.
     *
     * @return the R-squared statistic
     */
    double getRSquared();

    /**
     * Gets \( R^2 \) adjusted for the number of parameters in the model.
     *
     * @return the adjusted R-squared statistic
     */
    double getAdjustedRSquared();

    /**
     * Gets the estimated variance of the error term \( \epsilon \): the sum of squared
     * residuals divided by the residual degrees of freedom.
     *
     * @return the error variance estimate
     */
    double getErrorVariance();

    /**
     * Gets the standard error of the regression, the square root of the
     * {@linkplain #getErrorVariance() error variance}.
     *
     * @return the standard error of the regression
     */
    double getRegressionStandardError();

    /**
     * Gets the two-sided p-value of the {@code t}-statistic for each regression
     * parameter, testing the null hypothesis that the parameter is zero. Indices
     * correspond to {@link #getCoefficients()}.
     *
     * <p>The statistic is the parameter estimate divided by its standard error, referred
     * to a {@code t}-distribution with degrees of freedom equal to the number of
     * observations minus the number of parameters.
     *
     * @return the p-values of the parameter estimates
     * @throws IllegalArgumentException if there are no residual degrees of freedom
     */
    double[] getCoefficientPValues();

    /**
     * Gets a \( 100(1 - \alpha) \) percent confidence interval for each regression
     * parameter, computed from the {@code t}-distribution with degrees of freedom equal
     * to the number of observations minus the number of parameters. Row {@code i} of the
     * result holds the lower and upper bound of the interval for the parameter at index
     * {@code i} of {@link #getCoefficients()}.
     *
     * @param alpha Significance level of the interval.
     * @return the confidence intervals of the parameter estimates
     * @throws IllegalArgumentException if {@code alpha} is not in the open interval
     * {@code (0, 1)}, or if there are no residual degrees of freedom
     */
    double[][] getCoefficientConfidenceIntervals(double alpha);

    /**
     * Predicts the response for the given values of the regressor variables using the
     * estimated parameters. The intercept, if fitted, is applied automatically and is
     * not part of the input.
     *
     * @param x Values of the regressor variables.
     * @return the predicted response
     * @throws IllegalArgumentException if the number of values does not match the number
     * of regressor variables in the model
     */
    double predict(double... x);
}
