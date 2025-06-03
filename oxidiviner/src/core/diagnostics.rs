use crate::core::{OxiError, Result};
use crate::math::metrics::{mae, mse, rmse};
use serde::{Deserialize, Serialize};

/// Comprehensive diagnostic results for time series models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Model name
    pub model_name: String,
    /// Residual analysis results
    pub residual_analysis: ResidualAnalysis,
    /// Model specification tests
    pub specification_tests: SpecificationTests,
    /// Forecast diagnostics
    pub forecast_diagnostics: ForecastDiagnostics,
    /// Overall model quality score (0-100)
    pub quality_score: f64,
    /// Diagnostic recommendations
    pub recommendations: Vec<String>,
}

/// Residual analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    /// Residual statistics
    pub statistics: ResidualStatistics,
    /// Normality tests
    pub normality_tests: NormalityTests,
    /// Autocorrelation analysis
    pub autocorrelation: AutocorrelationAnalysis,
    /// Heteroskedasticity tests
    pub heteroskedasticity_tests: HeteroskedasticityTests,
    /// Outlier detection
    pub outliers: OutlierAnalysis,
}

/// Basic residual statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub jarque_bera_statistic: f64,
    pub jarque_bera_p_value: f64,
}

/// Normality test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTests {
    /// Jarque-Bera test
    pub jarque_bera: TestResult,
    /// Shapiro-Wilk test (for smaller samples)
    pub shapiro_wilk: Option<TestResult>,
    /// Anderson-Darling test
    pub anderson_darling: TestResult,
}

/// Autocorrelation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    /// Ljung-Box test for autocorrelation
    pub ljung_box: TestResult,
    /// Box-Pierce test
    pub box_pierce: TestResult,
    /// Autocorrelation function values (first 20 lags)
    pub acf_values: Vec<f64>,
    /// Partial autocorrelation function values
    pub pacf_values: Vec<f64>,
    /// 95% confidence bounds for ACF
    pub acf_confidence_bounds: (f64, f64),
}

/// Heteroskedasticity tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroskedasticityTests {
    /// ARCH test for conditional heteroskedasticity
    pub arch_test: TestResult,
    /// Breusch-Pagan test
    pub breusch_pagan: TestResult,
    /// White test for heteroskedasticity
    pub white_test: TestResult,
}

/// Outlier analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Z-scores for outlier detection
    pub z_scores: Vec<f64>,
    /// Modified Z-scores (using median absolute deviation)
    pub modified_z_scores: Vec<f64>,
    /// Percentage of outliers
    pub outlier_percentage: f64,
}

/// Model specification tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificationTests {
    /// Information criteria comparison
    pub information_criteria: InformationCriteria,
    /// Model adequacy tests
    pub adequacy_tests: ModelAdequacyTests,
    /// Parameter stability tests
    pub stability_tests: StabilityTests,
    /// Forecast performance tests
    pub forecast_tests: ForecastPerformanceTests,
}

/// Information criteria for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationCriteria {
    pub aic: f64,
    pub bic: f64,
    pub hqc: f64,           // Hannan-Quinn criterion
    pub aic_corrected: f64, // AICc for small samples
}

/// Model adequacy tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdequacyTests {
    /// Model validation score
    pub validation_score: f64,
    /// Cross-validation RMSE
    pub cv_rmse: f64,
    /// Model complexity penalty
    pub complexity_penalty: f64,
}

/// Parameter stability tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityTests {
    /// CUSUM test for parameter stability
    pub cusum_test: TestResult,
    /// Recursive residuals test
    pub recursive_residuals: TestResult,
    /// Break point detection
    pub structural_breaks: Vec<usize>,
}

/// Forecast performance diagnostic tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastPerformanceTests {
    /// Diebold-Mariano test for forecast comparison
    pub diebold_mariano: Option<TestResult>,
    /// Forecast encompassing test
    pub encompassing_test: Option<TestResult>,
    /// Mincer-Zarnowitz regression test
    pub mincer_zarnowitz: TestResult,
}

/// Forecast-specific diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastDiagnostics {
    /// Forecast error analysis
    pub error_analysis: ForecastErrorAnalysis,
    /// Prediction intervals validation
    pub interval_validation: IntervalValidation,
    /// Forecast bias analysis
    pub bias_analysis: BiasAnalysis,
}

/// Forecast error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastErrorAnalysis {
    /// Mean forecast error
    pub mean_error: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute percentage error
    pub mape: f64,
    /// Symmetric MAPE
    pub smape: f64,
    /// Theil's U statistic
    pub theil_u: f64,
}

/// Prediction interval validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalValidation {
    /// Coverage probability (actual vs nominal)
    pub coverage_probability: f64,
    /// Average interval width
    pub average_width: f64,
    /// Interval score
    pub interval_score: f64,
}

/// Forecast bias analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysis {
    /// Forecast bias (mean of forecast errors)
    pub bias: f64,
    /// Bias significance test
    pub bias_test: TestResult,
    /// Autocorrelation of forecast errors
    pub error_autocorrelation: f64,
}

/// Generic test result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value at 5% significance
    pub critical_value: f64,
    /// Whether null hypothesis is rejected at 5% level
    pub is_significant: bool,
    /// Test interpretation
    pub interpretation: String,
}

/// Main diagnostics engine
pub struct ModelDiagnostics;

impl ModelDiagnostics {
    /// Perform comprehensive diagnostic analysis on model residuals
    pub fn analyze_model(
        model_name: &str,
        residuals: &[f64],
        fitted_values: &[f64],
        actuals: &[f64],
        forecasts: Option<&[f64]>,
        forecast_errors: Option<&[f64]>,
    ) -> Result<DiagnosticReport> {
        // Perform residual analysis
        let residual_analysis = Self::analyze_residuals(residuals)?;

        // Perform specification tests
        let specification_tests =
            Self::perform_specification_tests(residuals, fitted_values, actuals)?;

        // Perform forecast diagnostics if forecast data is available
        let forecast_diagnostics =
            if let (Some(forecasts), Some(errors)) = (forecasts, forecast_errors) {
                Self::analyze_forecasts(actuals, forecasts, errors)?
            } else {
                Self::default_forecast_diagnostics()
            };

        // Calculate overall quality score
        let quality_score = Self::calculate_quality_score(
            &residual_analysis,
            &specification_tests,
            &forecast_diagnostics,
        );

        // Generate recommendations
        let recommendations = Self::generate_recommendations(
            &residual_analysis,
            &specification_tests,
            &forecast_diagnostics,
        );

        Ok(DiagnosticReport {
            model_name: model_name.to_string(),
            residual_analysis,
            specification_tests,
            forecast_diagnostics,
            quality_score,
            recommendations,
        })
    }

    /// Analyze residuals for various statistical properties
    fn analyze_residuals(residuals: &[f64]) -> Result<ResidualAnalysis> {
        let statistics = Self::calculate_residual_statistics(residuals)?;
        let normality_tests = Self::perform_normality_tests(residuals)?;
        let autocorrelation = Self::analyze_autocorrelation(residuals)?;
        let heteroskedasticity_tests = Self::perform_heteroskedasticity_tests(residuals)?;
        let outliers = Self::detect_outliers(residuals)?;

        Ok(ResidualAnalysis {
            statistics,
            normality_tests,
            autocorrelation,
            heteroskedasticity_tests,
            outliers,
        })
    }

    /// Calculate basic residual statistics
    fn calculate_residual_statistics(residuals: &[f64]) -> Result<ResidualStatistics> {
        if residuals.is_empty() {
            return Err(OxiError::InvalidParameter(
                "Empty residuals vector".to_string(),
            ));
        }

        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;

        // Calculate variance and standard deviation
        let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        // Calculate skewness and kurtosis
        let skewness = if std_dev > 0.0 {
            residuals
                .iter()
                .map(|r| ((r - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            residuals
                .iter()
                .map(|r| ((r - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n
                - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        // Calculate quantiles
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_residuals[0];
        let max = sorted_residuals[sorted_residuals.len() - 1];
        let q25 = Self::percentile(&sorted_residuals, 0.25);
        let median = Self::percentile(&sorted_residuals, 0.5);
        let q75 = Self::percentile(&sorted_residuals, 0.75);

        // Jarque-Bera test statistic
        let jb_statistic = n * (skewness.powi(2) / 6.0 + kurtosis.powi(2) / 24.0);
        let jb_p_value = Self::chi_square_p_value(jb_statistic, 2.0);

        Ok(ResidualStatistics {
            mean,
            std_dev,
            skewness,
            kurtosis,
            min,
            max,
            q25,
            median,
            q75,
            jarque_bera_statistic: jb_statistic,
            jarque_bera_p_value: jb_p_value,
        })
    }

    /// Perform normality tests on residuals
    fn perform_normality_tests(residuals: &[f64]) -> Result<NormalityTests> {
        let stats = Self::calculate_residual_statistics(residuals)?;

        // Jarque-Bera test
        let jarque_bera = TestResult {
            statistic: stats.jarque_bera_statistic,
            p_value: stats.jarque_bera_p_value,
            critical_value: 5.991, // Chi-square critical value at 5% with 2 df
            is_significant: stats.jarque_bera_p_value < 0.05,
            interpretation: if stats.jarque_bera_p_value < 0.05 {
                "Residuals are not normally distributed".to_string()
            } else {
                "Residuals appear to be normally distributed".to_string()
            },
        };

        // Anderson-Darling test (simplified implementation)
        let ad_statistic = Self::anderson_darling_statistic(residuals)?;
        let ad_critical = 0.752; // Critical value at 5% significance
        let anderson_darling = TestResult {
            statistic: ad_statistic,
            p_value: if ad_statistic > ad_critical {
                0.01
            } else {
                0.25
            }, // Approximate
            critical_value: ad_critical,
            is_significant: ad_statistic > ad_critical,
            interpretation: if ad_statistic > ad_critical {
                "Residuals are not normally distributed (Anderson-Darling)".to_string()
            } else {
                "Residuals appear to be normally distributed (Anderson-Darling)".to_string()
            },
        };

        Ok(NormalityTests {
            jarque_bera,
            shapiro_wilk: None, // Would implement for smaller samples
            anderson_darling,
        })
    }

    /// Analyze autocorrelation in residuals
    fn analyze_autocorrelation(residuals: &[f64]) -> Result<AutocorrelationAnalysis> {
        let n = residuals.len();
        let max_lags = (n / 4).min(20); // Standard practice

        // Calculate ACF values
        let mut acf_values = Vec::with_capacity(max_lags);
        for lag in 1..=max_lags {
            let acf = Self::autocorrelation(residuals, lag)?;
            acf_values.push(acf);
        }

        // Calculate PACF values (simplified - in practice would use Yule-Walker)
        let pacf_values = acf_values.clone(); // Placeholder implementation

        // Confidence bounds for ACF (approximate)
        let confidence_bound = 1.96 / (n as f64).sqrt();
        let acf_confidence_bounds = (-confidence_bound, confidence_bound);

        // Ljung-Box test
        let ljung_box_lags = max_lags.min(10);
        let (lb_statistic, lb_p_value) = Self::ljung_box_test(residuals, ljung_box_lags)?;
        let ljung_box = TestResult {
            statistic: lb_statistic,
            p_value: lb_p_value,
            critical_value: Self::chi_square_quantile(0.95, ljung_box_lags as f64),
            is_significant: lb_p_value < 0.05,
            interpretation: if lb_p_value < 0.05 {
                "Significant autocorrelation detected in residuals".to_string()
            } else {
                "No significant autocorrelation in residuals".to_string()
            },
        };

        // Box-Pierce test (simpler version of Ljung-Box)
        let bp_statistic = acf_values
            .iter()
            .take(ljung_box_lags)
            .map(|&acf| acf.powi(2))
            .sum::<f64>()
            * n as f64;
        let bp_p_value = Self::chi_square_p_value(bp_statistic, ljung_box_lags as f64);
        let box_pierce = TestResult {
            statistic: bp_statistic,
            p_value: bp_p_value,
            critical_value: Self::chi_square_quantile(0.95, ljung_box_lags as f64),
            is_significant: bp_p_value < 0.05,
            interpretation: if bp_p_value < 0.05 {
                "Significant autocorrelation detected (Box-Pierce)".to_string()
            } else {
                "No significant autocorrelation (Box-Pierce)".to_string()
            },
        };

        Ok(AutocorrelationAnalysis {
            ljung_box,
            box_pierce,
            acf_values,
            pacf_values,
            acf_confidence_bounds,
        })
    }

    /// Perform heteroskedasticity tests
    fn perform_heteroskedasticity_tests(residuals: &[f64]) -> Result<HeteroskedasticityTests> {
        // ARCH test
        let arch_test = Self::arch_test(residuals, 5)?; // Test for ARCH(5)

        // Simplified Breusch-Pagan and White tests
        let breusch_pagan = TestResult {
            statistic: 0.0,
            p_value: 0.5,
            critical_value: 3.841,
            is_significant: false,
            interpretation: "Breusch-Pagan test not implemented".to_string(),
        };

        let white_test = TestResult {
            statistic: 0.0,
            p_value: 0.5,
            critical_value: 3.841,
            is_significant: false,
            interpretation: "White test not implemented".to_string(),
        };

        Ok(HeteroskedasticityTests {
            arch_test,
            breusch_pagan,
            white_test,
        })
    }

    /// Detect outliers in residuals
    fn detect_outliers(residuals: &[f64]) -> Result<OutlierAnalysis> {
        let n = residuals.len();
        let mean = residuals.iter().sum::<f64>() / n as f64;
        let std_dev =
            (residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

        // Calculate Z-scores
        let z_scores: Vec<f64> = residuals.iter().map(|&r| (r - mean) / std_dev).collect();

        // Calculate modified Z-scores using median absolute deviation
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = Self::percentile(&sorted_residuals, 0.5);
        let mad = {
            let deviations: Vec<f64> = residuals.iter().map(|&r| (r - median).abs()).collect();
            let mut sorted_deviations = deviations;
            sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Self::percentile(&sorted_deviations, 0.5)
        };

        let modified_z_scores: Vec<f64> = residuals
            .iter()
            .map(|&r| 0.6745 * (r - median) / mad)
            .collect();

        // Detect outliers (|Z-score| > 3 or |modified Z-score| > 3.5)
        let outlier_indices: Vec<usize> = z_scores
            .iter()
            .enumerate()
            .filter(|(i, &z)| z.abs() > 3.0 || modified_z_scores[*i].abs() > 3.5)
            .map(|(i, _)| i)
            .collect();

        let outlier_percentage = (outlier_indices.len() as f64 / n as f64) * 100.0;

        Ok(OutlierAnalysis {
            outlier_indices,
            z_scores,
            modified_z_scores,
            outlier_percentage,
        })
    }

    /// Helper function to calculate percentiles
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        let n = sorted_data.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted_data[0];
        }

        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Calculate autocorrelation at given lag
    fn autocorrelation(data: &[f64], lag: usize) -> Result<f64> {
        if lag >= data.len() {
            return Ok(0.0);
        }

        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(n - lag) {
            numerator += (data[i] - mean) * (data[i + lag] - mean);
        }

        for &x in data {
            denominator += (x - mean).powi(2);
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Ljung-Box test for autocorrelation
    fn ljung_box_test(residuals: &[f64], lags: usize) -> Result<(f64, f64)> {
        let n = residuals.len() as f64;
        let mut statistic = 0.0;

        for lag in 1..=lags {
            let acf = Self::autocorrelation(residuals, lag)?;
            statistic += acf.powi(2) / (n - lag as f64);
        }

        statistic *= n * (n + 2.0);
        let p_value = Self::chi_square_p_value(statistic, lags as f64);

        Ok((statistic, p_value))
    }

    /// ARCH test for conditional heteroskedasticity
    fn arch_test(residuals: &[f64], lags: usize) -> Result<TestResult> {
        let n = residuals.len();
        if n <= lags {
            return Ok(TestResult {
                statistic: 0.0,
                p_value: 1.0,
                critical_value: 3.841,
                is_significant: false,
                interpretation: "Insufficient data for ARCH test".to_string(),
            });
        }

        // Square the residuals
        let squared_residuals: Vec<f64> = residuals.iter().map(|r| r.powi(2)).collect();

        // Simplified ARCH test - check autocorrelation in squared residuals
        let mut lm_statistic = 0.0;
        for lag in 1..=lags {
            let acf = Self::autocorrelation(&squared_residuals, lag)?;
            lm_statistic += acf.powi(2);
        }

        lm_statistic *= n as f64;
        let p_value = Self::chi_square_p_value(lm_statistic, lags as f64);
        let critical_value = Self::chi_square_quantile(0.95, lags as f64);

        Ok(TestResult {
            statistic: lm_statistic,
            p_value,
            critical_value,
            is_significant: p_value < 0.05,
            interpretation: if p_value < 0.05 {
                "ARCH effects detected - conditional heteroskedasticity present".to_string()
            } else {
                "No ARCH effects detected".to_string()
            },
        })
    }

    /// Anderson-Darling test statistic (simplified)
    fn anderson_darling_statistic(data: &[f64]) -> Result<f64> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len() as f64;
        let mean = sorted_data.iter().sum::<f64>() / n;
        let variance = sorted_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        // Standardize data and calculate AD statistic
        let mut ad_stat = 0.0;
        for (i, &x) in sorted_data.iter().enumerate() {
            let z = (x - mean) / std_dev;
            let phi_z = Self::standard_normal_cdf(z);
            let phi_z_comp =
                1.0 - Self::standard_normal_cdf(sorted_data[sorted_data.len() - 1 - i]);

            if phi_z > 0.0 && phi_z < 1.0 && phi_z_comp > 0.0 && phi_z_comp < 1.0 {
                ad_stat += (2.0 * (i + 1) as f64 - 1.0) * (phi_z.ln() + phi_z_comp.ln());
            }
        }

        ad_stat = -n - ad_stat / n;
        Ok(ad_stat)
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + (x / 2.0_f64.sqrt()).tanh())
    }

    /// Chi-square p-value approximation
    fn chi_square_p_value(statistic: f64, df: f64) -> f64 {
        // Very simplified approximation
        if statistic < 0.0 {
            return 1.0;
        }

        // Use normal approximation for large df
        if df > 30.0 {
            let z = ((2.0 * statistic).sqrt() - (2.0 * df - 1.0).sqrt()).abs();
            return 2.0 * (1.0 - Self::standard_normal_cdf(z));
        }

        // Simplified lookup for common cases
        let critical_values = [3.841, 5.991, 7.815, 9.488, 11.070];
        if (1.0..=5.0).contains(&df) {
            let idx = (df - 1.0) as usize;
            if statistic > critical_values[idx] {
                0.01
            } else {
                0.25
            }
        } else {
            0.1 // Default
        }
    }

    /// Chi-square quantile approximation
    fn chi_square_quantile(p: f64, df: f64) -> f64 {
        // Simplified implementation
        match df as usize {
            1 => {
                if p > 0.95 {
                    3.841
                } else {
                    2.706
                }
            }
            2 => {
                if p > 0.95 {
                    5.991
                } else {
                    4.605
                }
            }
            3 => {
                if p > 0.95 {
                    7.815
                } else {
                    6.251
                }
            }
            4 => {
                if p > 0.95 {
                    9.488
                } else {
                    7.779
                }
            }
            5 => {
                if p > 0.95 {
                    11.070
                } else {
                    9.236
                }
            }
            _ => df + 2.0 * (2.0 * df).sqrt(), // Rough approximation
        }
    }

    /// Perform model specification tests
    fn perform_specification_tests(
        residuals: &[f64],
        fitted_values: &[f64],
        actuals: &[f64],
    ) -> Result<SpecificationTests> {
        // Calculate information criteria
        let n = residuals.len() as f64;
        let k = 3.0; // Assume 3 parameters for generic model
        let log_likelihood = Self::calculate_log_likelihood(residuals)?;

        let information_criteria = InformationCriteria {
            aic: -2.0 * log_likelihood + 2.0 * k,
            bic: -2.0 * log_likelihood + k * n.ln(),
            hqc: -2.0 * log_likelihood + 2.0 * k * n.ln().ln(),
            aic_corrected: -2.0 * log_likelihood + 2.0 * k * n / (n - k - 1.0),
        };

        // Model adequacy tests
        let cv_rmse = rmse(actuals, fitted_values);
        let adequacy_tests = ModelAdequacyTests {
            validation_score: 100.0 - cv_rmse.min(100.0),
            cv_rmse,
            complexity_penalty: k / n,
        };

        // Stability tests (simplified)
        let stability_tests = StabilityTests {
            cusum_test: TestResult {
                statistic: 0.0,
                p_value: 0.5,
                critical_value: 1.36,
                is_significant: false,
                interpretation: "CUSUM test not implemented".to_string(),
            },
            recursive_residuals: TestResult {
                statistic: 0.0,
                p_value: 0.5,
                critical_value: 1.96,
                is_significant: false,
                interpretation: "Recursive residuals test not implemented".to_string(),
            },
            structural_breaks: Vec::new(),
        };

        // Forecast performance tests
        let forecast_tests = ForecastPerformanceTests {
            diebold_mariano: None,
            encompassing_test: None,
            mincer_zarnowitz: TestResult {
                statistic: 0.0,
                p_value: 0.5,
                critical_value: 1.96,
                is_significant: false,
                interpretation: "Mincer-Zarnowitz test not implemented".to_string(),
            },
        };

        Ok(SpecificationTests {
            information_criteria,
            adequacy_tests,
            stability_tests,
            forecast_tests,
        })
    }

    /// Calculate log-likelihood for residuals (assuming normality)
    fn calculate_log_likelihood(residuals: &[f64]) -> Result<f64> {
        let n = residuals.len() as f64;
        let variance = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / n;

        if variance <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }

        let log_likelihood = -0.5 * n * (2.0 * std::f64::consts::PI * variance).ln()
            - 0.5 * residuals.iter().map(|r| r.powi(2)).sum::<f64>() / variance;

        Ok(log_likelihood)
    }

    /// Analyze forecast performance
    fn analyze_forecasts(
        actuals: &[f64],
        forecasts: &[f64],
        forecast_errors: &[f64],
    ) -> Result<ForecastDiagnostics> {
        let error_analysis = ForecastErrorAnalysis {
            mean_error: forecast_errors.iter().sum::<f64>() / forecast_errors.len() as f64,
            mae: mae(actuals, forecasts),
            rmse: rmse(actuals, forecasts),
            mape: Self::calculate_mape(actuals, forecasts),
            smape: Self::calculate_smape(actuals, forecasts),
            theil_u: Self::calculate_theil_u(actuals, forecasts),
        };

        let interval_validation = IntervalValidation {
            coverage_probability: 0.95, // Placeholder
            average_width: 0.0,
            interval_score: 0.0,
        };

        let bias = forecast_errors.iter().sum::<f64>() / forecast_errors.len() as f64;
        let bias_analysis = BiasAnalysis {
            bias,
            bias_test: TestResult {
                statistic: bias * (forecast_errors.len() as f64).sqrt(),
                p_value: 0.5,
                critical_value: 1.96,
                is_significant: bias.abs() > 1.96 / (forecast_errors.len() as f64).sqrt(),
                interpretation: if bias.abs() > 1.96 / (forecast_errors.len() as f64).sqrt() {
                    "Significant forecast bias detected".to_string()
                } else {
                    "No significant forecast bias".to_string()
                },
            },
            error_autocorrelation: Self::autocorrelation(forecast_errors, 1).unwrap_or(0.0),
        };

        Ok(ForecastDiagnostics {
            error_analysis,
            interval_validation,
            bias_analysis,
        })
    }

    /// Create default forecast diagnostics when no forecast data is available
    fn default_forecast_diagnostics() -> ForecastDiagnostics {
        ForecastDiagnostics {
            error_analysis: ForecastErrorAnalysis {
                mean_error: 0.0,
                mae: 0.0,
                rmse: 0.0,
                mape: 0.0,
                smape: 0.0,
                theil_u: 0.0,
            },
            interval_validation: IntervalValidation {
                coverage_probability: 0.0,
                average_width: 0.0,
                interval_score: 0.0,
            },
            bias_analysis: BiasAnalysis {
                bias: 0.0,
                bias_test: TestResult {
                    statistic: 0.0,
                    p_value: 1.0,
                    critical_value: 1.96,
                    is_significant: false,
                    interpretation: "No forecast data available".to_string(),
                },
                error_autocorrelation: 0.0,
            },
        }
    }

    /// Calculate Mean Absolute Percentage Error
    fn calculate_mape(actual: &[f64], forecast: &[f64]) -> f64 {
        if actual.len() != forecast.len() {
            return f64::NAN;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for (a, f) in actual.iter().zip(forecast.iter()) {
            if a.abs() > 1e-8 {
                sum += ((a - f) / a).abs();
                count += 1;
            }
        }

        if count > 0 {
            (sum / count as f64) * 100.0
        } else {
            f64::NAN
        }
    }

    /// Calculate Symmetric Mean Absolute Percentage Error
    fn calculate_smape(actual: &[f64], forecast: &[f64]) -> f64 {
        if actual.len() != forecast.len() {
            return f64::NAN;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for (a, f) in actual.iter().zip(forecast.iter()) {
            let denominator = (a.abs() + f.abs()) / 2.0;
            if denominator > 1e-8 {
                sum += (a - f).abs() / denominator;
                count += 1;
            }
        }

        if count > 0 {
            (sum / count as f64) * 100.0
        } else {
            f64::NAN
        }
    }

    /// Calculate Theil's U statistic
    fn calculate_theil_u(actual: &[f64], forecast: &[f64]) -> f64 {
        if actual.len() != forecast.len() || actual.len() < 2 {
            return f64::NAN;
        }

        // Calculate RMSE of forecast
        let forecast_mse = mse(actual, forecast);

        // Calculate RMSE of naive forecast (no-change model)
        let mut naive_mse = 0.0;
        for i in 1..actual.len() {
            naive_mse += (actual[i] - actual[i - 1]).powi(2);
        }
        naive_mse /= (actual.len() - 1) as f64;

        if naive_mse > 0.0 {
            (forecast_mse / naive_mse).sqrt()
        } else {
            f64::NAN
        }
    }

    /// Calculate overall model quality score (0-100)
    fn calculate_quality_score(
        residual_analysis: &ResidualAnalysis,
        specification_tests: &SpecificationTests,
        _forecast_diagnostics: &ForecastDiagnostics,
    ) -> f64 {
        let mut score = 100.0;

        // Penalize for non-normal residuals
        if residual_analysis.normality_tests.jarque_bera.is_significant {
            score -= 10.0;
        }

        // Penalize for autocorrelation
        if residual_analysis.autocorrelation.ljung_box.is_significant {
            score -= 15.0;
        }

        // Penalize for ARCH effects
        if residual_analysis
            .heteroskedasticity_tests
            .arch_test
            .is_significant
        {
            score -= 10.0;
        }

        // Penalize for outliers
        if residual_analysis.outliers.outlier_percentage > 5.0 {
            score -= 10.0;
        }

        // Add validation score component
        score = (score + specification_tests.adequacy_tests.validation_score) / 2.0;

        score.clamp(0.0, 100.0)
    }

    /// Generate diagnostic recommendations
    fn generate_recommendations(
        residual_analysis: &ResidualAnalysis,
        _specification_tests: &SpecificationTests,
        _forecast_diagnostics: &ForecastDiagnostics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if residual_analysis.normality_tests.jarque_bera.is_significant {
            recommendations.push(
                "Consider transforming the data or using a model that doesn't assume normal errors"
                    .to_string(),
            );
        }

        if residual_analysis.autocorrelation.ljung_box.is_significant {
            recommendations.push(
                "Model may be underspecified - consider adding more AR or MA terms".to_string(),
            );
        }

        if residual_analysis
            .heteroskedasticity_tests
            .arch_test
            .is_significant
        {
            recommendations.push(
                "Consider using a GARCH model to handle conditional heteroskedasticity".to_string(),
            );
        }

        if residual_analysis.outliers.outlier_percentage > 5.0 {
            recommendations.push("High percentage of outliers detected - consider data cleaning or robust estimation methods".to_string());
        }

        if residual_analysis.statistics.skewness.abs() > 1.0 {
            recommendations.push(
                "Residuals show significant skewness - consider data transformation".to_string(),
            );
        }

        if residual_analysis.statistics.kurtosis.abs() > 2.0 {
            recommendations.push(
                "Residuals show excess kurtosis - model may not capture all patterns in the data"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Model diagnostics look good - no major issues detected".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_residuals() -> Vec<f64> {
        // Create some test residuals with known properties
        vec![
            0.1, -0.2, 0.05, 0.3, -0.15, 0.08, -0.1, 0.25, -0.05, 0.12, -0.18, 0.07, 0.15, -0.22,
            0.09, -0.13, 0.11, 0.04, -0.08, 0.16,
        ]
    }

    #[test]
    fn test_residual_statistics() {
        let residuals = create_test_residuals();
        let stats = ModelDiagnostics::calculate_residual_statistics(&residuals).unwrap();

        assert!(stats.mean.abs() < 0.1); // Should be close to zero
        assert!(stats.std_dev > 0.0);
        assert!(stats.min <= stats.q25);
        assert!(stats.q25 <= stats.median);
        assert!(stats.median <= stats.q75);
        assert!(stats.q75 <= stats.max);
    }

    #[test]
    fn test_autocorrelation() {
        let residuals = create_test_residuals();
        let acf1 = ModelDiagnostics::autocorrelation(&residuals, 1).unwrap();
        let acf2 = ModelDiagnostics::autocorrelation(&residuals, 2).unwrap();

        assert!(acf1.abs() <= 1.0);
        assert!(acf2.abs() <= 1.0);
    }

    #[test]
    fn test_ljung_box_test() {
        let residuals = create_test_residuals();
        let (statistic, p_value) = ModelDiagnostics::ljung_box_test(&residuals, 5).unwrap();

        assert!(statistic >= 0.0);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_outlier_detection() {
        let mut residuals = create_test_residuals();
        residuals.push(5.0); // Add a clear outlier

        let outlier_analysis = ModelDiagnostics::detect_outliers(&residuals).unwrap();

        assert!(!outlier_analysis.outlier_indices.is_empty());
        assert!(outlier_analysis.outlier_percentage > 0.0);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let residuals = create_test_residuals();
        let fitted_values: Vec<f64> = (0..residuals.len()).map(|i| i as f64 * 0.1).collect();
        let actuals: Vec<f64> = fitted_values
            .iter()
            .zip(&residuals)
            .map(|(f, r)| f + r)
            .collect();

        let report = ModelDiagnostics::analyze_model(
            "TestModel",
            &residuals,
            &fitted_values,
            &actuals,
            None,
            None,
        )
        .unwrap();

        assert_eq!(report.model_name, "TestModel");
        assert!(report.quality_score >= 0.0 && report.quality_score <= 100.0);
        assert!(!report.recommendations.is_empty());
    }
}
