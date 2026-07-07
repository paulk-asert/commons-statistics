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
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Test cases for {@link SimpleLinearRegression}.
 */
class SimpleLinearRegressionTest {
    /**
     * NIST "Norris" certified reference data set.
     * Each row is {y, x}.
     *
     * @see <a href="https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Norris.dat">Norris</a>
     */
    private static final double[][] NORRIS = {
        {0.1, 0.2}, {338.8, 337.4}, {118.1, 118.2},
        {888.0, 884.6}, {9.2, 10.1}, {228.1, 226.5}, {668.5, 666.3}, {998.5, 996.3},
        {449.1, 448.6}, {778.9, 777.0}, {559.2, 558.2}, {0.3, 0.4}, {0.1, 0.6}, {778.1, 775.5},
        {668.8, 666.9}, {339.3, 338.0}, {448.9, 447.5}, {10.8, 11.6}, {557.7, 556.0},
        {228.3, 228.1}, {998.0, 995.8}, {888.8, 887.6}, {119.6, 120.2}, {0.3, 0.3},
        {0.6, 0.3}, {557.6, 556.8}, {339.3, 339.1}, {888.0, 887.2}, {998.5, 999.0},
        {778.9, 779.0}, {10.2, 11.1}, {117.6, 118.3}, {228.9, 229.2}, {668.4, 669.1},
        {449.2, 448.9}, {0.2, 0.5}
    };

    @Test
    void testNorris() {
        final SimpleLinearRegression regression = SimpleLinearRegression.create();
        for (final double[] pair : NORRIS) {
            regression.accept(pair[1], pair[0]);
        }
        Assertions.assertEquals(36, regression.getN());
        final RegressionResult r = regression.getResult();
        final double[] beta = r.getCoefficients();
        final double[] errors = r.getCoefficientStandardErrors();
        // Certified values from NIST
        Assertions.assertEquals(1.00211681802045, beta[1], 1e-11, "slope");
        Assertions.assertEquals(0.429796848199937E-03, errors[1], 1e-11, "slope std err");
        Assertions.assertEquals(-0.262323073774029, beta[0], 1e-11, "intercept");
        Assertions.assertEquals(0.232818234301152, errors[0], 1e-11, "intercept std err");
        Assertions.assertEquals(0.999993745883712, r.getRSquared(), 1e-11, "r-square");
        // Residual mean square against R
        Assertions.assertEquals(0.782864662630069, r.getErrorVariance(), 1e-10, "MSE");
        Assertions.assertEquals(36, r.getN());
        Assertions.assertTrue(r.hasIntercept());
        // Prediction applies the intercept and slope
        Assertions.assertEquals(beta[0], r.predict(0), 0.0);
        Assertions.assertEquals(beta[0] + beta[1], r.predict(1), 1e-12);
    }

    @Test
    void testOfMatchesAccept() {
        final int n = NORRIS.length;
        final double[] x = new double[n];
        final double[] y = new double[n];
        final SimpleLinearRegression expected = SimpleLinearRegression.create();
        for (int i = 0; i < n; i++) {
            x[i] = NORRIS[i][1];
            y[i] = NORRIS[i][0];
            expected.accept(x[i], y[i]);
        }
        final RegressionResult e = expected.getResult();
        final RegressionResult r = SimpleLinearRegression.of(x, y).getResult();
        Assertions.assertArrayEquals(e.getCoefficients(), r.getCoefficients());
        Assertions.assertArrayEquals(e.getCoefficientStandardErrors(),
            r.getCoefficientStandardErrors());
        Assertions.assertEquals(e.getRSquared(), r.getRSquared());
    }

    @Test
    void testCombine() {
        final SimpleLinearRegression all = SimpleLinearRegression.create();
        final SimpleLinearRegression first = SimpleLinearRegression.create();
        final SimpleLinearRegression second = SimpleLinearRegression.create();
        final int split = NORRIS.length / 2;
        for (int i = 0; i < NORRIS.length; i++) {
            all.accept(NORRIS[i][1], NORRIS[i][0]);
            if (i < split) {
                first.accept(NORRIS[i][1], NORRIS[i][0]);
            } else {
                second.accept(NORRIS[i][1], NORRIS[i][0]);
            }
        }
        final SimpleLinearRegression combined = first.combine(second);
        Assertions.assertSame(first, combined);
        Assertions.assertEquals(all.getN(), combined.getN());
        final RegressionResult e = all.getResult();
        final RegressionResult r = combined.getResult();
        assertRelativelyEquals(e.getCoefficients()[0], r.getCoefficients()[0], 1e-10);
        assertRelativelyEquals(e.getCoefficients()[1], r.getCoefficients()[1], 1e-10);
        assertRelativelyEquals(e.getRSquared(), r.getRSquared(), 1e-10);
        assertRelativelyEquals(e.getErrorVariance(), r.getErrorVariance(), 1e-8);
    }

    @Test
    void testCombineEmpty() {
        final SimpleLinearRegression empty = SimpleLinearRegression.create();
        final SimpleLinearRegression data = SimpleLinearRegression.of(
            new double[] {1, 2, 3}, new double[] {2, 4, 7});
        final RegressionResult e = data.getResult();
        // Merging an empty accumulator is a no-op
        final RegressionResult r1 = data.combine(SimpleLinearRegression.create()).getResult();
        Assertions.assertArrayEquals(e.getCoefficients(), r1.getCoefficients());
        // Merging into an empty accumulator copies the state
        final RegressionResult r2 = empty.combine(data).getResult();
        Assertions.assertArrayEquals(e.getCoefficients(), r2.getCoefficients());
    }

    @Test
    void testVsCommonsMath3() {
        final SplittableRandom rng = new SplittableRandom(42);
        final SimpleRegression reference = new SimpleRegression();
        final SimpleLinearRegression regression = SimpleLinearRegression.create();
        for (int i = 0; i < 100; i++) {
            final double x = rng.nextDouble() * 10;
            final double y = 3 + 2 * x + rng.nextDouble();
            reference.addData(x, y);
            regression.accept(x, y);
        }
        final RegressionResult r = regression.getResult();
        assertRelativelyEquals(reference.getSlope(), r.getCoefficients()[1], 1e-10);
        assertRelativelyEquals(reference.getIntercept(), r.getCoefficients()[0], 1e-10);
        assertRelativelyEquals(reference.getSlopeStdErr(),
            r.getCoefficientStandardErrors()[1], 1e-10);
        assertRelativelyEquals(reference.getInterceptStdErr(),
            r.getCoefficientStandardErrors()[0], 1e-10);
        assertRelativelyEquals(reference.getRSquare(), r.getRSquared(), 1e-10);
        assertRelativelyEquals(reference.getMeanSquareError(), r.getErrorVariance(), 1e-10);
    }

    @Test
    void testInference() {
        // Data from Mendenhall and Sincich, example 10.3; expected values verified with R
        final double[][] data = {
            {15.6, 5.2}, {26.8, 6.1}, {37.8, 8.7}, {36.4, 8.5}, {35.5, 8.8},
            {18.6, 4.9}, {15.3, 4.5}, {7.9, 2.5}, {0.0, 1.1}
        };
        final SimpleLinearRegression regression = SimpleLinearRegression.create();
        for (final double[] pair : data) {
            regression.accept(pair[0], pair[1]);
        }
        final RegressionResult r = regression.getResult();
        // Significance of the slope
        Assertions.assertEquals(4.596e-07, r.getCoefficientPValues()[1], 1e-8);
        // Half-width of the default 95% slope confidence interval
        final double[] slopeCI = r.getCoefficientConfidenceIntervals(0.05)[1];
        Assertions.assertEquals(0.0270713794287, (slopeCI[1] - slopeCI[0]) / 2, 1e-8);
        Assertions.assertEquals(r.getCoefficients()[1], (slopeCI[1] + slopeCI[0]) / 2, 1e-12);
    }

    @Test
    void testInferenceBadFit() {
        // Data with a bad linear fit; expected values verified with R
        final double[][] data = {
            {1, 1}, {2, 0}, {3, 5}, {4, 2}, {5, -1}, {6, 12}
        };
        final SimpleLinearRegression regression = SimpleLinearRegression.create();
        for (final double[] pair : data) {
            regression.accept(pair[0], pair[1]);
        }
        final RegressionResult r = regression.getResult();
        Assertions.assertEquals(0.261829133982, r.getCoefficientPValues()[1], 1e-11);
        final double[] slopeCI = r.getCoefficientConfidenceIntervals(0.05)[1];
        Assertions.assertEquals(2.97802204827, (slopeCI[1] - slopeCI[0]) / 2, 1e-8);
        // A lower significance level widens the interval
        final double[] tighter = r.getCoefficientConfidenceIntervals(0.01)[1];
        Assertions.assertTrue(tighter[1] - tighter[0] > slopeCI[1] - slopeCI[0]);
    }

    @Test
    void testConfidenceIntervalInvalidAlpha() {
        final RegressionResult r = SimpleLinearRegression.of(
            new double[] {1, 2, 3, 4}, new double[] {2, 3, 5, 7}).getResult();
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> r.getCoefficientConfidenceIntervals(0));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> r.getCoefficientConfidenceIntervals(1));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> r.getCoefficientConfidenceIntervals(Double.NaN));
    }

    @Test
    void testInsufficientData() {
        final SimpleLinearRegression regression = SimpleLinearRegression.create();
        Assertions.assertEquals(Double.NaN, regression.getResult().getCoefficients()[1]);
        regression.accept(1, 2);
        Assertions.assertEquals(Double.NaN, regression.getResult().getCoefficients()[1]);
    }

    @Test
    void testOfLengthMismatch() {
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> SimpleLinearRegression.of(new double[3], new double[4]));
    }

    /**
     * Assert the values are equal to the given relative tolerance.
     *
     * @param expected Expected value.
     * @param actual Actual value.
     * @param eps Relative tolerance.
     */
    private static void assertRelativelyEquals(double expected, double actual, double eps) {
        Assertions.assertEquals(expected, actual, Math.abs(expected) * eps);
    }
}
