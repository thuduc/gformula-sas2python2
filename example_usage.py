"""
Example usage of the Python GFormula implementation

This script demonstrates how to use the converted GFormula class
to run parametric g-formula analyses.
"""

import pandas as pd
import numpy as np
from gformula import GFormula

# Example 1: Basic binary survival outcome analysis
def example_binary_survival():
    """Example with binary survival outcome"""
    
    # Create example data
    # In practice, you would load your actual data
    np.random.seed(42)
    n_subjects = 1000
    n_timepoints = 10
    
    # Generate example longitudinal data
    data_list = []
    for i in range(n_subjects):
        for t in range(n_timepoints):
            row = {
                'id': i,
                'time': t,
                'outcome': np.random.binomial(1, 0.1),  # Binary outcome
                'cov1': np.random.binomial(1, 0.3),     # Binary covariate
                'cov2': np.random.normal(0, 1),         # Continuous covariate
                'baseline_var': np.random.binomial(1, 0.5) if t == 0 else None
            }
            data_list.append(row)
    
    data = pd.DataFrame(data_list)
    # Forward fill baseline variable
    data['baseline_var'] = data.groupby('id')['baseline_var'].ffill()
    
    # Initialize GFormula
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome_var='outcome',
        outcome_type='binsurv',
        time_points=n_timepoints,
        
        # Fixed covariates
        fixedcov='baseline_var',
        
        # Time-varying covariates
        ncov=2,
        cov1='cov1',
        cov1otype=1,  # Binary
        cov1ptype='lag1',
        
        cov2='cov2',
        cov2otype=3,  # Continuous
        cov2ptype='lag1',
        
        # Bootstrap settings
        nsamples=10,  # Use more in practice (e.g., 500)
        seed=7834,
        
        # Output settings
        outputs='yes',
        print_stats=1
    )
    
    # Run analysis
    print("Running g-formula analysis...")
    results = gf.run()
    
    print("\nAnalysis complete!")
    print(f"Number of subjects: {data['id'].nunique()}")
    print(f"Number of time points: {n_timepoints}")
    print(f"Outcome events: {data['outcome'].sum()}")
    
    return results


# Example 2: End-of-follow-up continuous outcome
def example_continuous_eof():
    """Example with continuous end-of-follow-up outcome"""
    
    # Create example data
    np.random.seed(42)
    n_subjects = 500
    n_timepoints = 5
    
    data_list = []
    for i in range(n_subjects):
        for t in range(n_timepoints):
            row = {
                'id': i,
                'time': t,
                'treatment': np.random.binomial(1, 0.5) if t == 0 else None,
                'confounder': np.random.normal(0, 1),
                'outcome': np.random.normal(100, 15) if t == n_timepoints - 1 else None
            }
            data_list.append(row)
    
    data = pd.DataFrame(data_list)
    data['treatment'] = data.groupby('id')['treatment'].ffill()
    
    # Initialize GFormula for EOF analysis
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome_var='outcome',
        outcome_type='conteofu',  # Continuous EOF
        time_points=n_timepoints,
        
        # Fixed covariates
        fixedcov='treatment',
        
        # Time-varying covariates
        ncov=1,
        cov1='confounder',
        cov1otype=3,  # Continuous
        cov1ptype='lag1',
        
        # No bootstrap for this example
        nsamples=0,
        
        # Output settings
        outputs='yes'
    )
    
    # Run analysis
    print("\nRunning continuous EOF analysis...")
    results = gf.run()
    
    print("\nAnalysis complete!")
    print(f"Mean outcome: {data['outcome'].mean():.2f}")
    
    return results


# Example 3: Analysis with competing events and censoring
def example_competing_events():
    """Example with competing events and censoring"""
    
    # Create example data
    np.random.seed(42)
    n_subjects = 1000
    n_timepoints = 8
    
    data_list = []
    for i in range(n_subjects):
        outcome_time = np.random.exponential(10)
        compevent_time = np.random.exponential(15)
        censor_time = np.random.exponential(20)
        
        for t in range(n_timepoints):
            row = {
                'id': i,
                'time': t,
                'outcome': 1 if t >= outcome_time and t < compevent_time and t < censor_time else 0,
                'compevent': 1 if t >= compevent_time and t < outcome_time and t < censor_time else 0,
                'censor': 1 if t >= censor_time else 0,
                'cov1': np.random.binomial(1, 0.4),
                'cov2': np.random.normal(0, 1)
            }
            data_list.append(row)
    
    data = pd.DataFrame(data_list)
    
    # Initialize GFormula with competing events
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome_var='outcome',
        outcome_type='binsurv',
        time_points=n_timepoints,
        
        # Competing event
        compevent='compevent',
        
        # Censoring
        censor='censor',
        maxipw='p99',  # Truncate IPW at 99th percentile
        
        # Time-varying covariates
        ncov=2,
        cov1='cov1',
        cov1otype=1,
        cov1ptype='lag1',
        
        cov2='cov2',
        cov2otype=3,
        cov2ptype='lag1',
        
        # Bootstrap settings
        nsamples=10,
        
        # Output settings
        outputs='yes'
    )
    
    # Run analysis
    print("\nRunning competing events analysis...")
    results = gf.run()
    
    print("\nAnalysis complete!")
    print(f"Primary outcomes: {data['outcome'].sum()}")
    print(f"Competing events: {data['compevent'].sum()}")
    print(f"Censoring events: {data['censor'].sum()}")
    
    return results


# Example 4: Using advanced covariate types
def example_advanced_covariates():
    """Example with splines, categorical variables, and cumulative averages"""
    
    # Create example data
    np.random.seed(42)
    n_subjects = 500
    n_timepoints = 6
    
    data_list = []
    for i in range(n_subjects):
        for t in range(n_timepoints):
            row = {
                'id': i,
                'time': t,
                'outcome': np.random.binomial(1, 0.05),
                'continuous_cov': np.random.gamma(2, 2),  # For spline
                'categorical_cov': np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]),
                'binary_cov': np.random.binomial(1, 0.3)  # For cumulative average
            }
            data_list.append(row)
    
    data = pd.DataFrame(data_list)
    
    # Initialize GFormula with advanced covariate types
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome_var='outcome',
        outcome_type='binsurv',
        time_points=n_timepoints,
        
        # Time with spline
        time_ptype='rcspline',
        time_knots=3,  # 3 knots for time spline
        
        # Time-varying covariates
        ncov=3,
        
        # Continuous with restricted cubic spline
        cov1='continuous_cov',
        cov1otype=3,
        cov1ptype='lagspl',
        cov1knots=4,  # 4 knots for spline
        
        # Categorical variable
        cov2='categorical_cov',
        cov2otype=4,
        cov2ptype='lagcat',
        cov2knots='1 2',  # Cut points for categories
        
        # Binary with cumulative average
        cov3='binary_cov',
        cov3otype=1,
        cov3ptype='cumavg',
        
        # No bootstrap for this example
        nsamples=0,
        
        # Output settings
        outputs='yes'
    )
    
    # Run analysis
    print("\nRunning analysis with advanced covariates...")
    results = gf.run()
    
    print("\nAnalysis complete!")
    print("Covariate types used:")
    print("- Restricted cubic splines for time and continuous covariate")
    print("- Categorical covariate with 3 levels")
    print("- Cumulative average of binary covariate")
    
    return results


# Main execution
if __name__ == "__main__":
    print("GFormula Python Implementation Examples")
    print("=" * 50)
    
    # Run examples
    print("\n1. Binary Survival Example")
    print("-" * 30)
    results1 = example_binary_survival()
    
    print("\n2. Continuous End-of-Follow-up Example")
    print("-" * 30)
    results2 = example_continuous_eof()
    
    print("\n3. Competing Events Example")
    print("-" * 30)
    results3 = example_competing_events()
    
    print("\n4. Advanced Covariates Example")
    print("-" * 30)
    results4 = example_advanced_covariates()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    
    # The results objects contain:
    # - parameters: Estimated model parameters
    # - simulated_data: Data simulated under interventions
    # - risk_estimates: Risk estimates under different interventions
    # - bootstrap_results: Bootstrap confidence intervals