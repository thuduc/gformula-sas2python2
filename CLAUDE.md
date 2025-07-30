# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the GFORMULA SAS macro (gformula4.0.sas) that implements the parametric g-formula for causal inference. The goal is to convert this SAS macro to Python code.

## SAS Macro Structure

The main gformula4.0.sas file (401.5KB) contains:
- Main `%macro gformula()` with extensive parameters for configuring the g-formula analysis
- Supporting macros for data preparation, bootstrapping, and results generation
- Key macro components include:
  - `%macro dataprep` - data preparation
  - `%macro samples` - sampling procedures
  - `%macro parameters` - parameter estimation
  - `%macro interv` - intervention definitions
  - `%macro results` - results calculation
  - `%macro bootstrap_results` - bootstrap confidence intervals

## Key Parameters and Concepts

The g-formula macro handles:
- **Outcome types**: binary survival (binsurv), binary/categorical/continuous end-of-follow-up (bineofu, cateofu, conteofu)
- **Time-varying covariates**: Up to ncov covariates with various predictor types (lag, cumavg, rcspline)
- **Interventions**: Multiple intervention scenarios can be defined
- **Competing events and censoring**: Built-in handling for competing risks

## Python Conversion Considerations

When converting to Python:
1. The macro uses extensive array operations and data steps that should be converted to pandas/numpy operations
2. SAS PROC procedures should be mapped to appropriate Python statistical libraries (statsmodels, scikit-learn)
3. The macro's parameter validation and error handling should be preserved
4. Consider creating a class-based structure to encapsulate the g-formula implementation
5. The bootstrap functionality should utilize Python's multiprocessing capabilities for efficiency

## Development Commands

Since this is a conversion project without existing Python infrastructure:
- Use pandas for data manipulation (SAS data steps)
- Use statsmodels for regression models (PROC GENMOD, PROC LOGISTIC)
- Use numpy for array operations
- Consider pytest for testing the converted functions
- Use black for Python code formatting

## Testing Strategy

When converting:
1. Create unit tests for individual macro components
2. Use example datasets to validate results match between SAS and Python versions
3. Pay special attention to edge cases in time-varying covariate handling
4. Ensure intervention specifications produce equivalent results