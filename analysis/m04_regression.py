#!/usr/bin/env python
"""Model 4. Predicting daily alcohol intake from a liver enzyme panel.

    python analysis/m04_regression.py

The BUPA liver disorders data (Forsyth 1990) holds five blood test results and
a self-reported number of half-pint equivalents of alcoholic beverage drunk per
day. The number of drinks is the quantity the panel was collected to relate to,
and it is treated here as the regression target. The seventh released column,
`selector`, is excluded; `src/data.load_bupa` documents why and the schema gate
checks that it is absent from the predictors.

Three models are fitted so that a weak result can be attributed. Ordinary least
squares gives the coefficients and their intervals. The same fit on a
log-transformed target tests whether the target's skew is what limits it. A
random forest tests whether a nonlinear relation is being missed by both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy
import statsmodels.api as statsmodels_api
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, splits, utils  # noqa: E402

PREFIX = "m04."


def cross_validated_r2(estimator, features, target) -> tuple[float, float]:
    """Mean and standard deviation of the coefficient of determination.

    A single train and test split of 341 rows gives one number with a wide
    sampling distribution, so the training partition is also scored by
    five-fold cross-validation and both are recorded.
    """
    folds = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = cross_val_score(estimator, features, target, cv=folds, scoring="r2")
    return float(scores.mean()), float(scores.std(ddof=1))


def main() -> int:
    record = utils.Metrics()
    with utils.Timer(record, "m04"):
        frame, counts = data.load_bupa()
        record.update(counts, PREFIX)

        predictors = list(config.BUPA_PREDICTORS)
        target = config.BUPA_TARGET
        if target in predictors or config.BUPA_EXCLUDED in frame.columns:
            raise ValueError("the excluded column reached the model matrix")

        train, test = splits.row_split(frame)
        record.update(splits.report(train, test), PREFIX + "split.")

        x_train = train[predictors].to_numpy(dtype=float)
        x_test = test[predictors].to_numpy(dtype=float)
        y_train = train[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)

        # --- ordinary least squares -----------------------------------------
        fitted = statsmodels_api.OLS(
            y_train, statsmodels_api.add_constant(x_train)).fit()
        names = ["intercept"] + predictors
        intervals = fitted.conf_int()
        utils.write_table(
            [{"term": name,
              "estimate": round(float(fitted.params[position]), 6),
              "standard_error": round(float(fitted.bse[position]), 6),
              "t": round(float(fitted.tvalues[position]), 4),
              "p_value": float("{:.4g}".format(fitted.pvalues[position])),
              "ci_low": round(float(intervals[position][0]), 6),
              "ci_high": round(float(intervals[position][1]), 6)}
             for position, name in enumerate(names)],
            "m04_coefficients")

        record.set(PREFIX + "ols.f_statistic", float(fitted.fvalue))
        record.set(PREFIX + "ols.f_p_value",
                   float("{:.4g}".format(fitted.f_pvalue)))
        record.set(PREFIX + "ols.r2_train_insample", float(fitted.rsquared))
        record.set(PREFIX + "ols.r2_train_adjusted", float(fitted.rsquared_adj))
        record.set(PREFIX + "ols.condition_number", float(fitted.condition_number))
        # The residuals are tested for normality because the coefficient
        # intervals above assume it. A rejected test does not invalidate the
        # point estimates and does widen the true uncertainty on them.
        jarque_bera = statsmodels_api.stats.stattools.jarque_bera(fitted.resid)
        record.set(PREFIX + "ols.residual_jarque_bera", float(jarque_bera[0]))
        record.set(PREFIX + "ols.residual_jarque_bera_p",
                   float("{:.4g}".format(jarque_bera[1])))
        record.set(PREFIX + "ols.residual_skew", float(jarque_bera[2]))

        predicted_test = fitted.predict(statsmodels_api.add_constant(x_test))
        record.update(evaluate.regression(y_test, predicted_test),
                      PREFIX + "ols.test_")
        record.update(
            evaluate.regression(y_train,
                                fitted.predict(
                                    statsmodels_api.add_constant(x_train))),
            PREFIX + "ols.train_")

        # --- the same fit on a log-transformed target ------------------------
        # Nine subjects report zero drinks, so the transform is log(1 + y) and
        # the prediction is returned to the original scale before it is scored.
        # Scoring on the transformed scale would report a coefficient of
        # determination against a different target.
        log_fitted = statsmodels_api.OLS(
            numpy.log1p(y_train), statsmodels_api.add_constant(x_train)).fit()
        log_predicted = numpy.expm1(
            log_fitted.predict(statsmodels_api.add_constant(x_test)))
        record.update(evaluate.regression(y_test, log_predicted),
                      PREFIX + "log_ols.test_")
        record.set(PREFIX + "log_ols.r2_train_insample",
                   float(log_fitted.rsquared))

        # --- a nonlinear alternative ----------------------------------------
        forest = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                       random_state=config.SEED, n_jobs=-1)
        forest.fit(x_train, y_train)
        record.update(evaluate.regression(y_test, forest.predict(x_test)),
                      PREFIX + "forest.test_")
        utils.write_table(
            [{"term": name, "importance": round(float(value), 6)}
             for name, value in zip(predictors, forest.feature_importances_)],
            "m04_forest_importance")

        # --- cross-validated comparison on the training partition -----------
        # The held-out partition holds 86 rows, so one test score carries a
        # wide sampling distribution. Five-fold cross-validation inside the
        # training partition is recorded beside it as the more stable estimate.
        cross_validated = {
            "least squares": cross_validated_r2(LinearRegression(),
                                                x_train, y_train),
            "random forest": cross_validated_r2(
                RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                      random_state=config.SEED, n_jobs=-1),
                x_train, y_train),
        }
        rows = []
        for label, key in (("least squares", "ols"),
                           ("log least squares", "log_ols"),
                           ("random forest", "forest")):
            row = {"model": label}
            for name in ("r2", "rmse", "mae"):
                row[name] = round(
                    record.number(PREFIX + key + ".test_" + name), 6)
            if label in cross_validated:
                mean, deviation = cross_validated[label]
                row["cv_r2_mean"] = round(mean, 6)
                row["cv_r2_sd"] = round(deviation, 6)
                record.set(PREFIX + key + ".cv_r2_mean", mean)
                record.set(PREFIX + key + ".cv_r2_sd", deviation)
            else:
                row["cv_r2_mean"] = ""
                row["cv_r2_sd"] = ""
            rows.append(row)
        utils.write_table(rows, "m04_model_comparison",
                          columns=["model", "r2", "rmse", "mae",
                                   "cv_r2_mean", "cv_r2_sd"])

        # The observed and predicted pairs the figure is drawn from.
        utils.write_table(
            [{"observed": float(observed), "predicted": round(float(value), 6)}
             for observed, value in zip(y_test, predicted_test)],
            "m04_predictions")

    record.save()
    print("model 4: test R2 = {:.4f}, RMSE = {:.3f} drinks/day on {} rows".format(
        record.number(PREFIX + "ols.test_r2"),
        record.number(PREFIX + "ols.test_rmse"),
        int(record.number(PREFIX + "ols.test_rows"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
