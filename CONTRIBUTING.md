# Contributing

Contributions are welcome! Here are a few areas where help would be most valuable:

## Good First Issues

- [ ] Add more Turkish public holidays (bridge days, administrative leave)
- [ ] Support Prophet / statsforecast baselines
- [ ] Optuna hyperparameter tuning integration
- [ ] Confidence intervals via quantile regression

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests if applicable
4. Run the pipeline end-to-end to ensure nothing is broken:
   ```bash
   python iyzico_forecasting.py
   ```
5. Open a pull request with a clear description of what changed and why

## Code Style

- Follow PEP 8
- Keep all configuration at the top of the file (the `AYARLAR` / Settings block)
- Document any new feature columns with their leakage-safety guarantee
  (i.e., confirm the feature only uses data available at forecast time)

## Reporting Bugs

Please open a GitHub Issue with:
- Python version and OS
- Full error traceback
- A minimal reproducible example (synthetic data is fine)
