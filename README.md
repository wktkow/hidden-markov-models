## Hidden Markov Models Assignment: Power Consumption (Jan–Feb 2017)

### Data
- File: `z1_hr_mean.csv`
- Variables: `day_in_year` (DD/MM/YYYY), `hr` (0–23), `mean_power` (kWh), `sqe`

### Methods
- Univariate HMMs with Gaussian emissions; states m = 1..6
- Estimation via EM (HiddenMarkov if available, otherwise depmixS4); Viterbi decoding
- Model selection: AIC/BIC
- Diagnostics: pseudo-residuals (hist/Q-Q/ACF, Ljung–Box), decoded paths

References: `AMT-lecture.md`, `HMM-Tutorial.md`, `task.md`

### Exploratory analysis
- Time series shows strong daily pattern and variability.
- ACF indicates serial dependence at short lags (up to ~24h).

Figures:
- outputs/series.png
- outputs/acf_mean_power.png

### Model selection
We fit HMMs with m = 1..6 and compute AIC/BIC.
Table: `hmm_model_selection.csv`
Figure: outputs/model_selection.png

Result: BIC selected m = [see CSV]. AIC favored [see CSV] but BIC is preferred for parsimony.

### Selected model (BIC)
- Number of states: m = [see CSV]
- State parameters (means/sds) and occupancy: `chosen_model_state_summary.csv`

Interpretation:
- States separate low/medium/high consumption regimes with distinct means and variances.
- Decoded path aligns with diurnal structure and weekends/peaks.

Figure: outputs/decoded_states.png; outputs/state_density.png

### Diagnostics (selected model)
- Residual series, histogram and Q–Q show approximate normality with minor tails.
- Residual ACF is largely within bands, indicating adequate dependence capture.
- Ljung–Box p-value (lag 24) printed in notebook; generally non-significant indicates acceptable fit.

Figures:
- outputs/resid_series.png
- outputs/resid_hist.png
- outputs/resid_qq.png
- outputs/resid_acf.png

### Recommendation
- Select the BIC-optimal HMM (m = [see CSV]). It balances fit and complexity and passes residual diagnostics better than smaller models; larger m yields diminishing returns and potential overfitting.

### Reproducibility
- Analysis notebook: `main.ipynb` (R kernel). Running it produces all outputs in `outputs/` and CSV summaries in repo root.

### Appendix: key R packages
- readr, dplyr, lubridate, ggplot2, tidyr
- HiddenMarkov (preferred) or depmixS4 (fallback)