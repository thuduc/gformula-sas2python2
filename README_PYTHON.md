# GFormula Python Implementation

This is a Python conversion of the SAS GFORMULA macro for parametric g-formula causal inference with time-varying confounders.

## Overview

The parametric g-formula is a method for estimating causal effects in longitudinal studies with time-varying confounders. This Python implementation provides the same functionality as the original SAS macro.

## Installation

### Requirements

```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

### Basic Usage

```python
from gformula import GFormula

# Initialize the g-formula with your data
gf = GFormula(
    data=your_dataframe,
    id_var='subject_id',
    time_var='time',
    outcome_var='outcome',
    outcome_type='binsurv',  # Binary survival outcome
    ncov=2,  # Number of time-varying covariates
    cov1='covariate1',
    cov1otype=1,  # Binary
    cov1ptype='lag1',  # Lag-1 predictor
    cov2='covariate2',
    cov2otype=3,  # Continuous
    cov2ptype='lag1'
)

# Run the analysis
results = gf.run()
```

## Key Features

### Outcome Types
- `binsurv`: Binary survival outcome
- `bineofu`: Binary end-of-follow-up outcome
- `conteofu`: Continuous end-of-follow-up outcome
- `cateofu`: Categorical end-of-follow-up outcome
- `conteofu2`: Continuous EOF with truncated normal
- `conteofu3`: Continuous EOF with Tobit regression
- `conteofu4`: Continuous EOF with logistic-linear approach

### Covariate Types (otype)
- 0: Fixed/deterministic
- 1: Binary
- 2: Zero-inflated binary
- 3: Continuous (normal)
- 4: Categorical/multinomial
- 5: Bounded continuous
- 6: Zero-inflated normal
- 7: Truncated normal

### Predictor Types (ptype)
- `lag1`, `lag2`, `lag3`: Lagged values
- `cumavg`: Cumulative average
- `lag1cumavg`, `lag2cumavg`: Lagged cumulative average
- `lagspl`: Lagged with splines
- `lagcat`: Lagged categorical
- `rcspline`: Restricted cubic spline
- `tsswitch1`: Time-switch model

## Main Parameters

### Required Parameters
- `data`: pandas DataFrame with longitudinal data
- `id_var`: Subject identifier variable name
- `time_var`: Time variable name
- `outcome_var`: Outcome variable name

### Optional Parameters
- `outcome_type`: Type of outcome (default: 'binsurv')
- `time_points`: Number of time points
- `fixedcov`: Fixed/baseline covariates
- `ncov`: Number of time-varying covariates
- `compevent`: Competing event variable
- `censor`: Censoring variable
- `nsamples`: Number of bootstrap samples
- `seed`: Random seed
- `numint`: Number of interventions to simulate

### Covariate-specific Parameters
For each covariate i (1 to ncov):
- `covi`: Covariate name
- `coviotype`: Outcome type for covariate
- `coviptype`: Predictor type
- `coviknots`: Knots for splines/categories
- `coviinteract`: Interaction terms
- `coviskip`: Time points to skip
- `coviwherem`: Modeling conditions

## Examples

See `example_usage.py` for detailed examples including:
1. Basic binary survival analysis
2. Continuous end-of-follow-up outcomes
3. Competing events and censoring
4. Advanced covariate specifications

## Differences from SAS Version

1. **Data Structure**: Uses pandas DataFrames instead of SAS datasets
2. **Parameter Specification**: Uses Python function arguments instead of SAS macro parameters
3. **Model Fitting**: Uses statsmodels and scikit-learn instead of SAS procedures
4. **Where Conditions**: Uses pandas query syntax instead of SAS where statements
5. **Output**: Returns Python dictionaries and DataFrames instead of SAS datasets

## Current Limitations

This is a line-by-line conversion focusing on core functionality. Some advanced features may need additional implementation:
- Custom user macros (`modusermacro`, `simusermacro`)
- Complex intervention specifications
- Some specialized model types
- Graph generation

## Output Structure

The `run()` method returns a dictionary containing:
- `parameters`: Estimated model coefficients
- `simulated_data`: Simulated data under interventions
- `risk_estimates`: Risk estimates for each intervention
- `bootstrap_results`: Bootstrap confidence intervals

## References

Logan RW, Young JG, Taubman SL, et al. GFORMULA SAS Macro. 2024.

## License

This Python conversion maintains the same MIT License as the original SAS macro.