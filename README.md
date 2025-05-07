# OxiDiviner: The Fortune Teller for Your Time Series

**OxiDiviner, the fortune teller, can help you gaze into the future of your data.** By leveraging the power of statistical and component-based models, OxiDiviner aims to provide robust tools for understanding temporal patterns, forecasting future values, and making data-driven decisions.

Whether you're tracking business KPIs, analyzing scientific measurements, or monitoring system metrics, OxiDiviner seeks to equip you with the insights needed to navigate the currents of time.

## What is OxiDiviner?

OxiDiviner is a, 
1. comprehensive Rust toolkit for time series analysis and forecasting.
2. Rust library implementing a selection of powerful time series models
3. An educational project exploring time series models in Rust.

It is designed to be, "efficient, easy-to-use, and extensible", providing building blocks and complete solutions for your forecasting needs within the Rust ecosystem.

This project draws inspiration from a rich history of statistical methods and aims to make them accessible and performant.

## Features

* **Diverse Model Support:** Implementations of ETS, ARIMA family models, and insights from Prophet-style approaches." or "A flexible framework for building and comparing various time series models.
* **Rust Powered:** Leveraging Rust's performance, safety, and concurrency features for efficient computation.
* **Extensible Design:** Easily add custom models or components.
* **Data Preprocessing Utilities:** Tools for cleaning, transforming, and preparing time series data.
* **Clear API:** Designed for ease of integration into your Rust applications.

## Getting Started

# Statistical and Component-Based Models for Time Series Analysis and Forecasting

This document provides an overview of common models used for analyzing and forecasting time series data. These models leverage the temporal dependencies inherent in such data, employing various techniques to capture patterns like trend, seasonality, autocorrelation, and the impact of special events.

## 1. Exponential Smoothing (ETS) Models

These models produce forecasts based on weighted averages of past observations, with weights decreasing exponentially for older data. They explicitly model components like level, trend, and seasonality.

* **Simple/Single Exponential Smoothing (SES):** Models the *level* for data with no clear trend or seasonality.
* **Holt's Linear Trend Method (Double Exponential Smoothing):** Models *level* and *trend* for data with a trend but no seasonality.
* **Holt-Winters' Seasonal Method (Triple Exponential Smoothing):** Models *level*, *trend*, and *seasonality*.
    * *Additive Seasonality:* Assumes constant seasonal fluctuations.
    * *Multiplicative Seasonality:* Assumes seasonal fluctuations proportional to the level.
* **ETS Framework (Error, Trend, Seasonality):** A general state-space framework covering various combinations of error, trend, and seasonality types (None, Additive, Multiplicative, Damped).

## 2. ARIMA Family Models (Autoregressive Integrated Moving Average)

These models focus on capturing autocorrelations in stationary (or differenced to become stationary) data.

* **Autoregressive (AR) Models (`AR(p)`):** The current value depends linearly on `p` previous values.
* **Moving Average (MA) Models (`MA(q)`):** The current value depends linearly on `q` past random error terms.
* **Autoregressive Moving Average (ARMA) Models (`ARMA(p, q)`):** Combines AR and MA components for stationary series.
* **Autoregressive Integrated Moving Average (ARIMA) Models (`ARIMA(p, d, q)`):** Extends ARMA to non-stationary data using `d` degrees of differencing to achieve stationarity.
* **Seasonal ARIMA (SARIMA) Models (`SARIMA(p, d, q)(P, D, Q)m`):** Extends ARIMA for seasonality with seasonal orders `(P, D, Q)` and seasonal frequency `m`.
* **ARIMA/SARIMA with Exogenous Regressors (ARIMAX / SARIMAX):** Includes external predictor variables in the model.
* **Fractional ARIMA (ARFIMA / FARIMA):** Allows fractional differencing (`d` can be non-integer) for modeling long-range dependence (long memory).

## 3. Decomposition Methods

These methods break down a time series into its constituent components, typically trend, seasonality, and a remainder/residual term.

* **Classical Decomposition:** Simple additive/multiplicative separation using moving averages. Primarily useful for analysis.
* **STL Decomposition (Seasonal and Trend decomposition using Loess):** A versatile and robust decomposition method using locally weighted regression.
* **SEATS (Seasonal Extraction in ARIMA Time Series):** Primarily used for seasonal adjustment, often in conjunction with ARIMA models.
* **X-11 / X-12-ARIMA / X-13-ARIMA-SEATS:** Sophisticated decomposition and seasonal adjustment methods developed and used by statistical agencies.

## 4. Regression Models Adapted for Time Series

These models incorporate regression techniques while accounting for time series characteristics.

* **Time Series Regression:** Uses standard linear regression with time-based predictors (e.g., time trend, seasonal dummy variables) and potentially external variables. Standard assumptions (like independent errors) often need careful checking.
* **Dynamic Regression Models:** Combines regression with time series error structures (e.g., ARIMA errors, essentially forming ARIMAX/SARIMAX) to handle autocorrelation in the residuals.

## 5. State Space Models (SSM)

A highly general framework representing time series models using unobserved 'state' variables that evolve over time according to probabilistic rules. Many ETS and ARIMA models can be expressed in this form.

* **Kalman Filter:** The primary algorithm for estimation, signal extraction, and forecasting in linear Gaussian state-space models.
* **Structural Time Series Models (STSM):** Explicitly models components like level, trend, and seasonality as unobserved states within the SSM framework (e.g., Basic Structural Model - BSM). Often estimated using the Kalman Filter.

## 6. Volatility Models (Commonly used in Finance)

These models focus specifically on forecasting the changing variance (volatility) of a time series, rather than just its level.

* **ARCH (Autoregressive Conditional Heteroskedasticity):** Models current variance based on past squared error terms.
* **GARCH (Generalized ARCH):** Extends ARCH by also including past variances in the model for current variance. Many variants exist (e.g., EGARCH, GJR-GARCH).

## 7. Multivariate Time Series Models

Used for modeling and forecasting multiple interdependent time series simultaneously.

* **Vector Autoregression (VAR):** Multivariate extension of AR models where each variable depends on its own lags and the lags of other variables in the system.
* **Vector Autoregression Moving-Average (VARMA):** Multivariate extension of ARMA models.
* **Vector Error Correction Model (VECM):** A variant of VAR used specifically for cointegrated time series (non-stationary series that have a stable, long-run relationship).

## 8. Prophet (Developed by Meta/Facebook)

Prophet is a modern forecasting procedure designed for forecasting time series data based on an additive model where non-linear trends are fit along with yearly, weekly, and daily seasonality, plus holiday effects. While using statistical concepts, it's often considered distinct from the classical ETS or ARIMA families.

* **Modeling Approach:** Implemented as an additive regression model: `y(t) = g(t) + s(t) + h(t) + εt`, where `g(t)` is the trend, `s(t)` is seasonality, `h(t)` represents holiday effects, and `εt` is the error term.
* **Key Features:**
    * **Trend:** Models trend using either a piecewise linear model (detecting *changepoints* automatically) or a logistic growth model for saturating forecasts.
    * **Seasonality:** Uses Fourier series to model periodic effects (yearly, weekly, daily, or custom).
    * **Holidays/Events:** Easily incorporates customizable lists of past and future holidays or events with significant impact.
    * **Robustness:** Designed to handle outliers, missing data, and shifts in the trend gracefully.
    * **Ease of Use:** Offers an intuitive API (primarily in R and Python) designed for analysts, often requiring less manual tuning than models like ARIMA for good baseline results.

* **Usage Context:** Prophet is frequently used as a powerful baseline or **alternative** to traditional models like ARIMA or ETS, especially for business time series with strong seasonal patterns, multiple seasonality periods, and holiday effects. It's common practice to **compare its forecasting accuracy against traditional models** on specific datasets to select the best approach for a given forecasting task.

---

*Note: The selection of the most appropriate model (whether from the traditional statistical families like ETS/ARIMA or newer procedures like Prophet) depends heavily on the specific characteristics of the time series data, the forecast horizon, the presence of external factors, and the analyst's goals (e.g., accuracy vs. interpretability).*
