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

/**
 * Statistical regression.
 *
 * <p>Provides ordinary least squares (OLS) estimation of linear regression models.
 * Two entry points are provided:
 *
 * <ul>
 *   <li>{@link org.apache.commons.statistics.regression.SimpleLinearRegression}: bivariate
 *       regression of {@code y} on a single {@code x} variable. Data may be streamed through
 *       the accumulator using {@code accept(x, y)}; instances may be combined, supporting
 *       parallel accumulation. No input data is stored.</li>
 *   <li>{@link org.apache.commons.statistics.regression.OLSRegression}: multiple linear
 *       regression of {@code y} on {@code k} regressor variables, fitted from data arrays
 *       using a QR decomposition of the design matrix.</li>
 * </ul>
 *
 * <p>Both produce a {@link org.apache.commons.statistics.regression.RegressionResult}
 * containing the estimated coefficients and summary statistics of the fit.
 *
 * <p>All implementations operate on primitive {@code double} data and do not expose or
 * require a matrix abstraction; the linear algebra used internally is not part of the
 * public API.
 *
 * @since 1.4
 */
package org.apache.commons.statistics.regression;
