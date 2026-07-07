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

import java.util.SplittableRandom;
import org.apache.commons.math3.stat.regression.GLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Test cases for {@link OLSRegression}.
 */
class OLSRegressionTest {
    /**
     * NIST "Longley" certified reference data set.
     * Each row is one observation: {y, x1, x2, x3, x4, x5, x6}.
     *
     * @see <a href="https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat">Longley</a>
     */
    private static final double[][] LONGLEY = {
        {60323, 83.0, 234289, 2356, 1590, 107608, 1947},
        {61122, 88.5, 259426, 2325, 1456, 108632, 1948},
        {60171, 88.2, 258054, 3682, 1616, 109773, 1949},
        {61187, 89.5, 284599, 3351, 1650, 110929, 1950},
        {63221, 96.2, 328975, 2099, 3099, 112075, 1951},
        {63639, 98.1, 346999, 1932, 3594, 113270, 1952},
        {64989, 99.0, 365385, 1870, 3547, 115094, 1953},
        {63761, 100.0, 363112, 3578, 3350, 116219, 1954},
        {66019, 101.2, 397469, 2904, 3048, 117388, 1955},
        {67857, 104.6, 419180, 2822, 2857, 118734, 1956},
        {68169, 108.4, 442769, 2936, 2798, 120445, 1957},
        {66513, 110.8, 444546, 4681, 2637, 121950, 1958},
        {68655, 112.6, 482704, 3813, 2552, 123366, 1959},
        {69564, 114.2, 502601, 3931, 2514, 125368, 1960},
        {69331, 115.7, 518173, 4806, 2572, 127852, 1961},
        {70551, 116.9, 554894, 4007, 2827, 130081, 1962}
    };

    /**
     * Extract the regressor variables from the reference data.
     *
     * @return the x data
     */
    private static double[][] longleyX() {
        final double[][] x = new double[LONGLEY.length][];
        for (int i = 0; i < LONGLEY.length; i++) {
            final double[] row = new double[LONGLEY[i].length - 1];
            System.arraycopy(LONGLEY[i], 1, row, 0, row.length);
            x[i] = row;
        }
        return x;
    }

    /**
     * Extract the response variable from the reference data.
     *
     * @return the y data
     */
    private static double[] longleyY() {
        final double[] y = new double[LONGLEY.length];
        for (int i = 0; i < LONGLEY.length; i++) {
            y[i] = LONGLEY[i][0];
        }
        return y;
    }

    @Test
    void testLongley() {
        final OLSRegression.Result r = OLSRegression.withDefaults().fit(longleyX(), longleyY());

        // Certified beta values from NIST
        Assertions.assertArrayEquals(new double[] {
            -3482258.63459582, 15.0618722713733,
            -0.358191792925910E-01, -2.02022980381683,
            -1.03322686717359, -0.511041056535807E-01,
            1829.15146461355}, r.getCoefficients(), 2e-8);

        // Certified standard errors from NIST
        Assertions.assertArrayEquals(new double[] {
            890420.383607373, 84.9149257747669,
            0.334910077722432E-01, 0.488399681651699,
            0.214274163161675, 0.226073200069370,
            455.478499142212}, r.getCoefficientStandardErrors(), 1e-6);

        // Expected residuals from R
        Assertions.assertArrayEquals(new double[] {
            267.340029759711, -94.0139423988359, 46.28716775752924,
            -410.114621930906, 309.7145907602313, -249.3112153297231,
            -164.0489563956039, -13.18035686637081, 14.30477260005235,
            455.394094551857, -17.26892711483297, -39.0550425226967,
            -155.5499735953195, -85.6713080421283, 341.9315139607727,
            -206.7578251937366}, r.getResiduals(), 1e-8);

        // Expected statistics from R
        Assertions.assertEquals(304.8540735619638, r.getRegressionStandardError(), 1e-10);
        Assertions.assertEquals(0.995479004577296, r.getRSquared(), 1e-12);
        Assertions.assertEquals(0.992465007628826, r.getAdjustedRSquared(), 1e-12);

        Assertions.assertEquals(16, r.getN());
        Assertions.assertTrue(r.hasIntercept());

        // Prediction for an observation is the observed value minus the residual
        final double[] residuals = r.getResiduals();
        Assertions.assertEquals(LONGLEY[0][0] - residuals[0],
            r.predict(longleyX()[0]), 1e-7);
    }

    @Test
    void testLongleyNoIntercept() {
        final OLSRegression.Result r = OLSRegression.withDefaults()
            .withIntercept(false).fit(longleyX(), longleyY());

        // Expected beta values from R
        Assertions.assertArrayEquals(new double[] {
            -52.99357013868291, 0.07107319907358,
            -0.42346585566399, -0.57256866841929,
            -0.41420358884978, 48.41786562001326}, r.getCoefficients(), 1e-11);

        // Expected standard errors from R
        Assertions.assertArrayEquals(new double[] {
            129.54486693117232, 0.03016640003786,
            0.41773654056612, 0.27899087467676, 0.32128496193363,
            17.68948737819961}, r.getCoefficientStandardErrors(), 1e-11);

        // Expected residuals from R
        Assertions.assertArrayEquals(new double[] {
            279.90274927293092, -130.32465380836874, 90.73228661967445, -401.31252201634948,
            -440.46768772620027, -543.54512853774793, 201.32111639536299, 215.90889365977932,
            73.09368242049943, 913.21694494481869, 424.82484953610174, -8.56475876776709,
            -361.32974610842876, 27.34560497213464, 151.28955976355002, -492.49937355336846},
            r.getResiduals(), 1e-10);

        // Expected statistics from R
        Assertions.assertEquals(475.1655079819517, r.getRegressionStandardError(), 1e-10);
        Assertions.assertEquals(0.9999670130706, r.getRSquared(), 1e-12);
        Assertions.assertEquals(0.999947220913, r.getAdjustedRSquared(), 1e-12);

        Assertions.assertFalse(r.hasIntercept());
    }

    @Test
    void testVsCommonsMath3() {
        final SplittableRandom rng = new SplittableRandom(42);
        final int n = 40;
        final int k = 3;
        final double[][] x = new double[n][k];
        final double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                x[i][j] = rng.nextDouble() * 10;
            }
            y[i] = 5 + 2 * x[i][0] - 3 * x[i][1] + 0.5 * x[i][2] + rng.nextDouble();
        }
        final OLSMultipleLinearRegression reference = new OLSMultipleLinearRegression();
        reference.newSampleData(y, x);

        final OLSRegression.Result r = OLSRegression.withDefaults().fit(x, y);
        assertRelativelyEquals(reference.estimateRegressionParameters(),
            r.getCoefficients(), 1e-8);
        assertRelativelyEquals(reference.estimateRegressionParametersStandardErrors(),
            r.getCoefficientStandardErrors(), 1e-8);
        assertRelativelyEquals(reference.estimateResiduals(), r.getResiduals(), 1e-6);
        Assertions.assertEquals(reference.calculateRSquared(), r.getRSquared(), 1e-12);
        Assertions.assertEquals(reference.calculateAdjustedRSquared(),
            r.getAdjustedRSquared(), 1e-12);
        Assertions.assertEquals(reference.estimateErrorVariance(), r.getErrorVariance(),
            reference.estimateErrorVariance() * 1e-10);
    }

    @Test
    void testInferenceVsCommonsMath3() {
        final SplittableRandom rng = new SplittableRandom(17);
        final int n = 30;
        final int k = 2;
        final double[][] x = new double[n][k];
        final double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                x[i][j] = rng.nextDouble() * 10;
            }
            y[i] = 1 + 4 * x[i][0] + rng.nextDouble() * 20;
        }
        final OLSMultipleLinearRegression reference = new OLSMultipleLinearRegression();
        reference.newSampleData(y, x);
        final double[] beta = reference.estimateRegressionParameters();
        final double[] se = reference.estimateRegressionParametersStandardErrors();
        // Reference inference statistics computed with the commons-math3 t-distribution
        final org.apache.commons.math3.distribution.TDistribution t =
            new org.apache.commons.math3.distribution.TDistribution(n - k - 1.0);

        final OLSRegression.Result r = OLSRegression.withDefaults().fit(x, y);
        final double[] p = r.getCoefficientPValues();
        final double alpha = 0.05;
        final double[][] ci = r.getCoefficientConfidenceIntervals(alpha);
        final double tc = t.inverseCumulativeProbability(1 - alpha / 2);
        for (int i = 0; i <= k; i++) {
            final double expectedP = 2 * (1 - t.cumulativeProbability(Math.abs(beta[i] / se[i])));
            Assertions.assertEquals(expectedP, p[i], Math.abs(expectedP) * 1e-6 + 1e-12, "p " + i);
            Assertions.assertEquals(beta[i] - tc * se[i], ci[i][0], Math.abs(beta[i]) * 1e-8, "lower " + i);
            Assertions.assertEquals(beta[i] + tc * se[i], ci[i][1], Math.abs(beta[i]) * 1e-8, "upper " + i);
        }
    }

    @Test
    void testLongleyPValues() {
        final OLSRegression.Result r = OLSRegression.withDefaults().fit(longleyX(), longleyY());
        final double[] p = r.getCoefficientPValues();
        // R: summary(lm(y ~ ., data=longley)) reports the model terms ordered as here.
        // Significant at 5%: intercept, x3 (unemployed), x4 (armed forces), x6 (year)
        final boolean[] significant = {true, false, false, true, true, false, true};
        for (int i = 0; i < p.length; i++) {
            Assertions.assertTrue(p[i] >= 0 && p[i] <= 1, "range " + i);
            Assertions.assertEquals(significant[i], p[i] < 0.05, "significance " + i);
        }
    }

    @Test
    void testLeverageVsCommonsMath3() {
        final double[][] x = longleyX();
        final double[] y = longleyY();
        final OLSMultipleLinearRegression reference = new OLSMultipleLinearRegression();
        reference.newSampleData(y, x);
        final double[][] hat = reference.calculateHat().getData();

        final double[] leverage = OLSRegression.withDefaults().fit(x, y).getLeverage();
        double sum = 0;
        for (int i = 0; i < leverage.length; i++) {
            Assertions.assertEquals(hat[i][i], leverage[i], 1e-8, "leverage " + i);
            sum += leverage[i];
        }
        // The leverage values sum to the number of parameters
        Assertions.assertEquals(7, sum, 1e-10);
    }

    @Test
    void testCoefficientCovariance() {
        final double[][] x = longleyX();
        final double[] y = longleyY();
        final OLSMultipleLinearRegression reference = new OLSMultipleLinearRegression();
        reference.newSampleData(y, x);
        // math3 returns the unscaled (X^T X)^-1
        final double[][] unscaled = reference.estimateRegressionParametersVariance();
        final double sigma2 = reference.estimateErrorVariance();

        final OLSRegression.Result r = OLSRegression.withDefaults().fit(x, y);
        final double[][] cov = r.getCoefficientCovariance();
        final double[] se = r.getCoefficientStandardErrors();
        for (int i = 0; i < cov.length; i++) {
            // The diagonal holds the squared standard errors
            Assertions.assertEquals(se[i] * se[i], cov[i][i], Math.abs(cov[i][i]) * 1e-12);
            for (int j = 0; j < cov.length; j++) {
                Assertions.assertEquals(cov[j][i], cov[i][j], "symmetry");
                Assertions.assertEquals(unscaled[i][j] * sigma2, cov[i][j],
                    Math.abs(cov[i][j]) * 1e-8, "covariance " + i + "," + j);
            }
        }
    }

    @Test
    void testWeightedUnitWeights() {
        final double[][] x = longleyX();
        final double[] y = longleyY();
        final double[] weights = new double[y.length];
        java.util.Arrays.fill(weights, 1);
        final OLSRegression.Result unweighted = OLSRegression.withDefaults().fit(x, y);
        final OLSRegression.Result weighted = OLSRegression.withDefaults().fit(x, y, weights);
        Assertions.assertArrayEquals(unweighted.getCoefficients(), weighted.getCoefficients());
        Assertions.assertArrayEquals(unweighted.getCoefficientStandardErrors(),
            weighted.getCoefficientStandardErrors());
        Assertions.assertArrayEquals(unweighted.getResiduals(), weighted.getResiduals());
        Assertions.assertArrayEquals(unweighted.getLeverage(), weighted.getLeverage());
        Assertions.assertEquals(unweighted.getRSquared(), weighted.getRSquared());
    }

    @Test
    void testWeightedScaleInvariance() {
        // Multiplying all weights by a constant does not change the fit or its statistics
        final double[][] x = longleyX();
        final double[] y = longleyY();
        final double[] weights = new double[y.length];
        java.util.Arrays.fill(weights, 3.5);
        final OLSRegression.Result unweighted = OLSRegression.withDefaults().fit(x, y);
        final OLSRegression.Result weighted = OLSRegression.withDefaults().fit(x, y, weights);
        assertRelativelyEquals(unweighted.getCoefficients(), weighted.getCoefficients(), 1e-10);
        assertRelativelyEquals(unweighted.getCoefficientStandardErrors(),
            weighted.getCoefficientStandardErrors(), 1e-10);
        Assertions.assertEquals(unweighted.getRSquared(), weighted.getRSquared(), 1e-12);
    }

    @Test
    void testWeightedReplication() {
        // An integer weight m is equivalent to including the observation m times
        // for the purpose of the parameter estimates
        final SplittableRandom rng = new SplittableRandom(3);
        final int n = 12;
        final double[][] x = new double[n][2];
        final double[] y = new double[n];
        final double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = rng.nextDouble() * 5;
            x[i][1] = rng.nextDouble() * 2;
            y[i] = 2 + 3 * x[i][0] - x[i][1] + rng.nextDouble();
            weights[i] = 1 + (i % 3);
        }
        // Duplicate each observation according to its weight
        final java.util.List<double[]> xDup = new java.util.ArrayList<>();
        final java.util.List<Double> yDup = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int m = 0; m < weights[i]; m++) {
                xDup.add(x[i]);
                yDup.add(y[i]);
            }
        }
        final double[][] x2 = xDup.toArray(new double[0][]);
        final double[] y2 = new double[yDup.size()];
        for (int i = 0; i < y2.length; i++) {
            y2[i] = yDup.get(i);
        }
        final double[] weightedBeta =
            OLSRegression.withDefaults().fit(x, y, weights).getCoefficients();
        final double[] duplicatedBeta =
            OLSRegression.withDefaults().fit(x2, y2).getCoefficients();
        assertRelativelyEquals(duplicatedBeta, weightedBeta, 1e-10);
    }

    @Test
    void testWeightedVsCommonsMath3GLS() {
        // WLS is GLS with a diagonal error covariance of 1/w
        final SplittableRandom rng = new SplittableRandom(7);
        final int n = 25;
        final int k = 3;
        final double[][] x = new double[n][k];
        final double[] y = new double[n];
        final double[] weights = new double[n];
        final double[][] omega = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                x[i][j] = rng.nextDouble() * 10;
            }
            y[i] = 5 + 2 * x[i][0] - 3 * x[i][1] + 0.5 * x[i][2] + rng.nextDouble() * 4;
            weights[i] = 0.5 + rng.nextDouble() * 2.5;
            omega[i][i] = 1 / weights[i];
        }
        final GLSMultipleLinearRegression reference = new GLSMultipleLinearRegression();
        reference.newSampleData(y, x, omega);

        final OLSRegression.Result r = OLSRegression.withDefaults().fit(x, y, weights);
        assertRelativelyEquals(reference.estimateRegressionParameters(),
            r.getCoefficients(), 1e-8);
        assertRelativelyEquals(reference.estimateRegressionParametersStandardErrors(),
            r.getCoefficientStandardErrors(), 1e-8);
        Assertions.assertEquals(reference.estimateErrorVariance(), r.getErrorVariance(),
            reference.estimateErrorVariance() * 1e-8);
        // The leverage values sum to the number of parameters in the weighted fit
        double sum = 0;
        for (final double h : r.getLeverage()) {
            sum += h;
        }
        Assertions.assertEquals(k + 1, sum, 1e-10);
    }

    @Test
    void testInvalidWeights() {
        final OLSRegression ols = OLSRegression.withDefaults();
        final double[][] x = new double[6][2];
        final double[] y = new double[6];
        final SplittableRandom rng = new SplittableRandom(5);
        for (int i = 0; i < x.length; i++) {
            x[i][0] = rng.nextDouble();
            x[i][1] = rng.nextDouble();
            y[i] = rng.nextDouble();
        }
        // Mismatched length
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> ols.fit(x, y, new double[5]));
        // Non-positive and non-finite weights
        for (final double bad : new double[] {0, -1, Double.NaN, Double.POSITIVE_INFINITY}) {
            final double[] weights = {1, 1, 1, bad, 1, 1};
            Assertions.assertThrows(IllegalArgumentException.class,
                () -> ols.fit(x, y, weights), "weight " + bad);
        }
    }

    @Test
    void testInvalidData() {
        final OLSRegression ols = OLSRegression.withDefaults();
        // No data
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> ols.fit(new double[0][0], new double[0]));
        // Mismatched lengths
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> ols.fit(new double[4][2], new double[5]));
        // Mismatched row length
        final double[][] ragged = {{1, 2}, {3}, {4, 5}, {6, 7}, {8, 9}};
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> ols.fit(ragged, new double[5]));
        // Not enough observations: n must exceed the number of parameters
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> ols.fit(new double[3][2], new double[3]));
    }

    @Test
    void testSingularMatrix() {
        // Second regressor is a multiple of the first
        final double[][] x = new double[6][2];
        final double[] y = new double[6];
        final SplittableRandom rng = new SplittableRandom(99);
        for (int i = 0; i < x.length; i++) {
            x[i][0] = rng.nextDouble();
            x[i][1] = 2 * x[i][0];
            y[i] = rng.nextDouble();
        }
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> OLSRegression.withDefaults().fit(x, y));
    }

    @Test
    void testWithIntercept() {
        final OLSRegression ols = OLSRegression.withDefaults();
        Assertions.assertSame(ols, ols.withIntercept(true));
        Assertions.assertSame(ols, ols.withIntercept(false).withIntercept(true));
    }

    /**
     * Assert the values are element-wise equal to the given relative tolerance.
     *
     * @param expected Expected values.
     * @param actual Actual values.
     * @param eps Relative tolerance.
     */
    private static void assertRelativelyEquals(double[] expected, double[] actual, double eps) {
        Assertions.assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i], Math.abs(expected[i]) * eps,
                "index " + i);
        }
    }
}
