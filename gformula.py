"""
GFORMULA Python Package

Authors: Roger W. Logan, Jessica G. Young, Sarah L. Taubman, Yu-Han Chiu, Emma McGee, 
Sally Picciotto, Goodarz Danaei, Miguel A. Hernán

Python conversion of the SAS GFORMULA macro for parametric g-formula causal inference.

Version: September 2024 (SAS) - Python conversion

Copyright (c) 2007, 2021, The President and Fellows of Harvard College

Permission is hereby granted, free of charge, to any person obtaining a copy of this 
software and associated documentation files (the "Software"), to deal in the Software 
without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons 
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

This software is provided under the standard MIT License:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import warnings
from dataclasses import dataclass, field
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial, Gaussian
from scipy import stats
import os
import sys
from copy import deepcopy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CovariateConfig:
    """Configuration for a single covariate in the g-formula"""
    name: str = ""
    otype: int = 1  # outcome type (default = binary)
    ptype: str = ""  # predictor type
    etype: str = ""  # predictor type for eof type outcome models
    mtype: str = "all"  # defines how variable is included in models
    cumint: str = ""  # covariate to include as factor in calculation of average
    skip: Union[int, List[int]] = -1  # time points where variable is not predicted
    inc: float = 0  # fixed increment for linearly increasing variables (type 0 only)
    knots: Optional[List[float]] = None  # knots for categories or splines
    eknots: Optional[List[float]] = None  # knots for eof models
    interact: Optional[List[str]] = None  # interaction terms
    wherem: str = "(1=1)"  # condition under which to model the covariate
    wherenosim: str = "(1=0)"  # condition under which not to simulate
    nosimelsemacro: Optional[Callable] = None  # user defined function for when wherenosim holds
    class_var: Optional[str] = None  # separate models by lagged covariate (type 4)
    classelse: Optional[Any] = None  # default value for class_var in simulation
    addvars: Optional[List[str]] = None  # additional variables as predictors
    genmacro: Optional[Callable] = None  # user defined function to generate variable
    modusermacro: Optional[Callable] = None  # user defined model fitting for otype=-1
    moddatausermacro: Optional[Callable] = None  # user defined data creation for otype=-1
    setblvar: Optional[Callable] = None  # user defined baseline variable setting
    simusermacro: Optional[Callable] = None  # user defined simulation for otype=-1
    barray: Optional[Any] = None  # extra array statements for coefficients
    sarray: Optional[Any] = None  # extra array statements for simulated variables
    randomvisitp: Optional[str] = None  # random visit process to model
    visitpmaxgap: float = 9e10  # maximum gap allowed between visits
    visitpwherem: str = "(1=1)"  # where conditions for visit process
    visitpcount: Optional[str] = None  # variable for initiating visit counter


class GFormula:
    """
    Main class for parametric g-formula implementation
    
    This class implements the parametric g-formula for causal inference
    with time-varying confounders.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 id_var: str,
                 time_var: str,
                 outcome_var: str,
                 outcome_type: str = "binsurv",
                 time_points: Optional[int] = None,
                 time_inc: int = 1,
                 interval: int = 1,
                 **kwargs):
        """
        Initialize GFormula object
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset for analysis
        id_var : str
            Unique identifier for subjects
        time_var : str
            Time point variable
        outcome_var : str
            Outcome variable (1=event, 0=no event for binary)
        outcome_type : str
            Type of outcome: 'binsurv', 'bineofu', 'cateofu', 'conteofu', 
            'conteofu2', 'conteofu3', 'conteofu4'
        time_points : int, optional
            Number of time points
        time_inc : int
            Time increment (default=1)
        interval : int
            Length of time between time points (default=1)
        **kwargs : additional parameters
        """
        
        # Core data parameters
        self.data = data.copy()
        self.id = id_var
        self.time = time_var
        self.outcome = outcome_var
        self.outcome_type = outcome_type
        
        # Time parameters
        self.time_points = time_points
        self.time_inc = time_inc
        self.interval = interval
        self.time_ptype = kwargs.get('time_ptype', None)
        self.time_knots = kwargs.get('time_knots', None)
        self.time_funcgen = kwargs.get('time_funcgen', None)
        
        # Outcome parameters
        self.outcome_interact = kwargs.get('outcome_interact', None)
        self.outcome_wherem = kwargs.get('outcome_wherem', "(1=1)")
        self.outcome_wherenosim = kwargs.get('outcome_wherenosim', "(1=0)")
        self.outcome_nosimelsemacro = kwargs.get('outcome_nosimelsemacro', None)
        self.use_history_eof = kwargs.get('use_history_eof', 0)
        
        # Competing event parameters
        self.compevent = kwargs.get('compevent', None)
        self.compevent_interact = kwargs.get('compevent_interact', None)
        self.compevent_wherem = kwargs.get('compevent_wherem', "(1=1)")
        self.compevent_wherenosim = kwargs.get('compevent_wherenosim', "(1=0)")
        self.compevent_nosimelsemacro = kwargs.get('compevent_nosimelsemacro', None)
        
        # Censoring parameters
        self.censor = kwargs.get('censor', None)
        self.maxipw = kwargs.get('maxipw', 'p99')
        self.censor_interact = kwargs.get('censor_interact', None)
        self.censor_wherem = kwargs.get('censor_wherem', "(1=1)")
        self.compevent_cens = kwargs.get('compevent_cens', 0)
        
        # Covariate parameters
        self.fixedcov = kwargs.get('fixedcov', None)
        self.ncov = kwargs.get('ncov', 0)
        
        # Initialize covariate configurations
        self.covariates = {}
        for i in range(1, 31):  # Support up to 30 covariates like SAS macro
            cov_name = kwargs.get(f'cov{i}', None)
            if cov_name:
                self.covariates[i] = CovariateConfig(
                    name=cov_name,
                    otype=kwargs.get(f'cov{i}otype', 1),
                    ptype=kwargs.get(f'cov{i}ptype', ''),
                    etype=kwargs.get(f'cov{i}etype', ''),
                    mtype=kwargs.get(f'cov{i}mtype', 'all'),
                    cumint=kwargs.get(f'cov{i}cumint', ''),
                    skip=kwargs.get(f'cov{i}skip', -1),
                    inc=kwargs.get(f'cov{i}inc', 0),
                    knots=kwargs.get(f'cov{i}knots', None),
                    eknots=kwargs.get(f'cov{i}eknots', None),
                    interact=kwargs.get(f'cov{i}interact', None),
                    wherem=kwargs.get(f'cov{i}wherem', "(1=1)"),
                    wherenosim=kwargs.get(f'cov{i}wherenosim', "(1=0)"),
                    nosimelsemacro=kwargs.get(f'cov{i}nosimelsemacro', None),
                    class_var=kwargs.get(f'cov{i}class', None),
                    classelse=kwargs.get(f'cov{i}classelse', None),
                    addvars=kwargs.get(f'cov{i}addvars', None),
                    genmacro=kwargs.get(f'cov{i}genmacro', None),
                    modusermacro=kwargs.get(f'cov{i}modusermacro', None),
                    moddatausermacro=kwargs.get(f'cov{i}moddatausermacro', None),
                    setblvar=kwargs.get(f'cov{i}setblvar', None),
                    simusermacro=kwargs.get(f'cov{i}simusermacro', None),
                    barray=kwargs.get(f'cov{i}barray', None),
                    sarray=kwargs.get(f'cov{i}sarray', None),
                    randomvisitp=kwargs.get(f'cov{i}randomvisitp', None),
                    visitpmaxgap=kwargs.get(f'cov{i}visitpmaxgap', 9e10),
                    visitpwherem=kwargs.get(f'cov{i}visitpwherem', "(1=1)"),
                    visitpcount=kwargs.get(f'cov{i}visitpcount', None)
                )
        
        # Other override options
        self.wherevars = kwargs.get('wherevars', None)
        self.keepsimuldata = kwargs.get('keepsimuldata', None)
        self.equalitiessimuldata = kwargs.get('equalitiessimuldata', None)
        self.eventaddvars = kwargs.get('eventaddvars', None)
        self.compeventaddvars = kwargs.get('compeventaddvars', None)
        self.censoraddvars = kwargs.get('censoraddvars', None)
        
        # Data storage parameters
        self.usebetadata = kwargs.get('usebetadata', 0)
        self.betadata = kwargs.get('betadata', None)
        self.simuldata = kwargs.get('simuldata', None)
        self.resultsdata = kwargs.get('resultsdata', None)
        self.survdata = kwargs.get('survdata', None)
        
        # Output parameters
        self.outputs = kwargs.get('outputs', 'yes')
        self.print_stats = kwargs.get('print_stats', 1)
        self.check_cov_models = kwargs.get('check_cov_models', 0)
        self.print_cov_means = kwargs.get('print_cov_means', 0)
        self.covmeandata = kwargs.get('covmeandata', None)
        self.save_raw_covmean = kwargs.get('save_raw_covmean', 0)
        self.observed_surv = kwargs.get('observed_surv', None)
        self.intervname = kwargs.get('intervname', None)
        
        # Simulation parameters
        self.refint = kwargs.get('refint', 0)
        self.seed = kwargs.get('seed', 7834)
        self.nsamples = kwargs.get('nsamples', 50)
        self.nsimul = kwargs.get('nsimul', None)
        self.nparam = kwargs.get('nparam', None)
        
        # Hazard ratio parameters
        self.hazardratio = kwargs.get('hazardratio', 0)
        self.intcomp = kwargs.get('intcomp', None)
        self.bootstrap_hazard = kwargs.get('bootstrap_hazard', 0)
        self.hazardname = kwargs.get('hazardname', None)
        
        # Intervention parameters
        self.numint = kwargs.get('numint', 0)
        self.sim_trunc = kwargs.get('sim_trunc', 1)
        
        # Bootstrap parameters
        self.sample_start = kwargs.get('sample_start', 0)
        self.sample_end = kwargs.get('sample_end', -1)
        self.savelib = kwargs.get('savelib', 'work')
        
        # Graph parameters
        self.rungraphs = kwargs.get('rungraphs', 0)
        self.title1 = kwargs.get('title1', None)
        self.title2 = kwargs.get('title2', None)
        self.title3 = kwargs.get('title3', None)
        self.titledata = kwargs.get('titledata', None)
        self.graphfile = kwargs.get('graphfile', 'gfile.pdf')
        self.tsize = kwargs.get('tsize', 1)
        
        # Other parameters
        self.runnc = kwargs.get('runnc', 1)
        self.weight = kwargs.get('weight', None)
        self.printlogstats = kwargs.get('printlogstats', 1)
        self.checkaddvars = kwargs.get('checkaddvars', 1)
        self.minimalistic = kwargs.get('minimalistic', 'no').lower()
        self.testing = kwargs.get('testing', 'no').lower()
        
        # Initialize local variables
        self._initialize_local_vars()
        
        # Set random seed
        np.random.seed(self.seed)
        
    def _initialize_local_vars(self):
        """Initialize local variables used in the macro"""
        
        # Set default values if not provided
        if self.time_points is None:
            self.time_points = self.data[self.time].max()
            
        if self.nsimul is None:
            self.nsimul = self.data[self.id].nunique()
            
        if self.nparam is None:
            self.nparam = self.data[self.id].nunique()
            
        if self.sample_end == -1:
            self.sample_end = self.nsamples
            
        # Initialize tracking variables
        self.anytsswitch1 = 0
        self.anylagcumavg = 0
        self.anyecumavg = 0
        self.anyfindpknots = 0
        self.anyfindeknots = 0
        
        # Create covariate list
        self.covlist = [cov.name for cov in self.covariates.values() if cov.name]
        
        # Initialize empty lists for tracking
        self.created_global = []
        
        # Handle minimalistic and testing conversions
        if self.minimalistic == '1':
            self.minimalistic = 'yes'
        elif self.minimalistic == '0':
            self.minimalistic = 'no'
            
        # Check for competing event censoring
        if not self.compevent:
            self.compevent_cens = 0
            
    def _process_parameters(self):
        """Process and validate parameters (equivalent to SAS macro parameter processing)"""
        
        # Handle testing mode
        if self.testing == 'yes':
            self.nsamples = 0
            self.minimalistic = 'no'
            
        # Handle bootstrap hazard
        if self.nsamples == 0:
            self.bootstrap_hazard = 0
            
        # Handle non-survival outcome types
        if self.outcome_type.upper() != 'BINSURV':
            self.hazardratio = None
            self.bootstrap_hazard = 0
            self.intcomp = None
            self.hazardname = None
            if self.censor and self.compevent:
                self.compevent_cens = 1
        
        # For survival outcomes, don't use history for EOF
        if self.outcome_type.upper() == 'BINSURV':
            self.use_history_eof = 0
            
        # Handle minimalistic mode
        if self.minimalistic == 'yes':
            self.rungraphs = 0
            self.print_cov_means = 0
            self.check_cov_models = 0
            
        # Validate required parameters
        if self.data is None:
            raise ValueError("ERROR: The data argument must be specified.")
            
        if self.runnc == 0 and self.refint == 0:
            raise ValueError("ERROR: When not running the natural course analysis, refint must be different from 0")
            
        # Initialize label use flags
        self.uselabelo = 0
        self.uselabelc = 0
        
        # Create time covariate (cov0)
        self._create_cov0_vars()
        
        # Process each covariate
        self._process_covariates()
        
        # Create fixed covariate information
        self._create_fixed_cov()
        
    def _create_cov0_vars(self):
        """Create time variable as covariate 0"""
        # Add time as covariate 0
        self.covariates[0] = CovariateConfig(
            name=self.time,
            otype=1,
            ptype=self.time_ptype or '',
            mtype='all',
            skip=-1,
            inc=0,
            knots=self.time_knots,
            wherem="(1=1)",
            wherenosim="(1=0)"
        )
        
    def _process_covariates(self):
        """Process covariate configurations"""
        
        for i in range(0, self.ncov + 1):
            if i not in self.covariates:
                continue
                
            cov = self.covariates[i]
            
            # Clean ptype and etype
            cov.ptype = cov.ptype.strip().lower() if cov.ptype else ''
            cov.etype = cov.etype.strip().lower() if cov.etype else ''
            
            # Initialize covariate-specific variables
            cov.simstart = 1
            cov.else_var = f"{cov.name}_b"
            cov.infixed = 0
            cov.findknots = 0
            cov.findeknots = 0
            
            # Process spline knots
            if 'spl' in cov.ptype.upper() or cov.ptype == 'tsswitch1':
                if not cov.knots or (isinstance(cov.knots, str) and cov.knots.upper() == 'NONE'):
                    cov.knots = 0
                    
                if isinstance(cov.knots, (int, float)):
                    if cov.knots == 1:
                        cov.knots = 3
                    elif cov.knots == 2:
                        cov.knots = 4
                    if cov.knots in (3, 4, 5):
                        cov.findknots = 1
                        self.anyfindpknots = 1
                        
                    if cov.knots == 0 and 'spl' in cov.ptype.upper():
                        raise ValueError(f"ERROR: WHEN USING COVARIATES OF TYPE LAGSPL OR SKPSPL, USER MUST DEFINE COV{i}KNOTS TO BE NON-EMPTY OR NON-ZERO")
                        
            # Process categorical knots
            if 'cat' in cov.ptype.upper():
                if not cov.knots or (isinstance(cov.knots, str) and cov.knots.upper() == 'NONE'):
                    raise ValueError(f"ERROR: WHEN USING COVARIATES OF TYPE -CAT, USER MUST DEFINE COV{i}KNOTS TO BE NON-EMPTY")
                    
            # Validate otype 0 increment
            if cov.otype == 0 and not cov.inc:
                raise ValueError("ERROR: Otype 0 increment must be specified")
                
            # Handle visit process
            cov.usevisitp = 0
            if cov.randomvisitp:
                cov.usevisitp = 1
                cov.visitpelse = f"{cov.randomvisitp}_b"
                
            # Validate mtype
            if cov.mtype.upper() not in ('ALL', 'SOME', 'NOCHECK'):
                cov.mtype = 'all'
                
            # Process etype for history EOF
            if self.use_history_eof == 1 and i > 0:
                self._process_etype(cov, i)
                
            # Handle special ptypes
            if cov.ptype.upper() == 'TSSWITCH1':
                self.anytsswitch1 = 1
                
            if cov.ptype.upper() in ('CUMAVG', 'LAG1CUMAVG', 'LAG2CUMAVG'):
                self._process_cumavg(cov, i)
                
    def _process_etype(self, cov, i):
        """Process etype for end-of-follow-up models"""
        if cov.etype:
            # Split etype into two parts
            etype_parts = cov.etype.split()
            cov.etype_part1 = etype_parts[0] if etype_parts else ''
            cov.etype_part2 = etype_parts[1] if len(etype_parts) > 1 else ''
            
            # Handle NONE values
            if cov.etype_part1.upper() == 'NONE' or cov.etype_part2.upper() == 'NONE' or cov.etype_part2 == '0':
                cov.etype_part1 = 'NONE'
                cov.etype_part2 = '0'
                
            cov.etype = cov.etype_part1
            
            # Handle tsswitch1 conversion
            if cov.ptype == 'tsswitch1' and cov.etype in ('tsswitch1', 'cumsum', 'cumsumnew'):
                if cov.etype_part1 == 'tsswitch1':
                    cov.etype_part1 = 'cumsum'
                    cov.etype = 'cumsum'
                    logger.info(f"CHANGING cov{i}etype FROM TSSWITCH1 TO CUMSUM.")
                    
            # Handle cumulative etypes
            if cov.etype.lower() in ('cumsum', 'cumsumnew', 'cumavg', 'cumavgnew'):
                if not cov.eknots or (isinstance(cov.eknots, str) and cov.eknots.upper() == 'NONE'):
                    cov.eknots = 0
                    
                if isinstance(cov.eknots, (int, float)):
                    if cov.eknots == 1:
                        cov.eknots = 3
                    elif cov.eknots == 2:
                        cov.eknots = 4
                    if cov.eknots in (3, 4, 5):
                        cov.findeknots = 1
                        self.anyfindeknots = 1
                        
            # Handle categorical etypes
            if 'cat' in cov.ptype:
                if not cov.eknots or (isinstance(cov.eknots, str) and cov.eknots.upper() == 'NONE'):
                    raise ValueError(f"ERROR: WHEN USING COVARIATES OF ETYPE -CAT, USER MUST DEFINE COV{i}EKNOTS TO BE NON-EMPTY")
                    
            # Set default etype_part2
            if not cov.etype_part2:
                cov.etype_part2 = str(self.time_points)
            if cov.etype_part2.upper() == 'ALL':
                cov.etype_part2 = str(self.time_points)
                
            # Track cumulative etypes
            if cov.etype.upper() in ('CUMSUM', 'CUMSUMNEW', 'CUMAVG', 'CUMAVGNEW'):
                if isinstance(cov.eknots, (int, float)) and cov.eknots > 0:
                    cov.findeknots = 1
                self.anyecumavg = 1
                
            if cov.etype.upper() in ('CUMSUMCAT', 'CUMSUMCATNEW', 'CUMAVGCAT', 'CUMAVGCATNEW'):
                if not cov.eknots or (isinstance(cov.eknots, str) and cov.eknots.upper() == 'NONE'):
                    raise ValueError(f"ERROR: WHEN USING AN ETYPE IN (CUMSUMCAT CUMSUMCATNEW CUMAVGCAT CUMAVGCATNEW) REQUIRES THE SETTING OF COV{i}EKNOTS")
                self.anyecumcat = 1
        else:
            cov.etype = ''
            cov.eknots = 0
            
    def _process_cumavg(self, cov, i):
        """Process cumulative average covariates"""
        self.anylagcumavg = 1
        
        if not cov.knots or (isinstance(cov.knots, str) and cov.knots.upper() == 'NONE'):
            cov.knots = 0
            
        if isinstance(cov.knots, list) and len(cov.knots) > 1:
            cov.lev = len(cov.knots)
            cov.cumavg_l1_knots = cov.knots
        else:
            if isinstance(cov.knots, (int, float)):
                if cov.knots == 1:
                    cov.knots = 3
                elif cov.knots == 2:
                    cov.knots = 4
                cov.lev = cov.knots
                self.anyfindpknots = 1
                if cov.knots > 0:
                    cov.findknots = 1
                    
        if cov.lev == 0:
            cov.lev = 2
            
        if self.printlogstats == 1:
            logger.info(f"Number of levels of {cov.name}: cov{i}lev={cov.lev}")
            
    def _create_fixed_cov(self):
        """Process fixed covariates (equivalent to SAS %createfixedcov macro)"""
        
        # Get number of fixed covariates
        nfixed = self._numargs(self.fixedcov)
        fixedcov1 = []
        
        if self.fixedcov:
            fixed_list = self.fixedcov if isinstance(self.fixedcov, list) else self.fixedcov.split()
            
            for i, word in enumerate(fixed_list):
                incovlist = False
                
                # Check if this fixed covariate is also a modeled covariate
                for j, cov in self.covariates.items():
                    if j == 0:
                        continue
                    if cov.name and word.upper() == cov.name.upper():
                        incovlist = True
                        cov.infixed = 1
                        
                        # Add appropriate interaction terms based on ptype
                        ptype_prefix = cov.ptype[:4].upper() if len(cov.ptype) >= 4 else cov.ptype.upper()
                        
                        if ptype_prefix in ('LAG1', 'CUMA', 'RCUM'):
                            if cov.interact is None:
                                cov.interact = []
                            elif isinstance(cov.interact, str):
                                cov.interact = cov.interact.split()
                            cov.interact.append(f"B{i+1}*L2")
                        elif ptype_prefix == 'LAG2':
                            if cov.interact is None:
                                cov.interact = []
                            elif isinstance(cov.interact, str):
                                cov.interact = cov.interact.split()
                            cov.interact.append(f"B{i+1}*L3")
                        elif ptype_prefix == 'LAG3':
                            if cov.interact is None:
                                cov.interact = []
                            elif isinstance(cov.interact, str):
                                cov.interact = cov.interact.split()
                            cov.interact.append(f"B{i+1}*L4")
                
                if incovlist:
                    fixedcov1.append(f"{word}_b")
                else:
                    fixedcov1.append(word)
        
        # Update fixedcov with processed list
        self.fixedcov = fixedcov1
        
        # Set fixedcov for each covariate
        for i, cov in self.covariates.items():
            if i == 0:
                continue
            if cov.infixed == 0:
                cov.fixedcov = self.fixedcov
            else:
                # For covariates that are in fixed list, exclude them from their own model
                cov.fixedcov = []
                for word in self.fixedcov:
                    if word.upper() != f"{cov.name.upper()}_B":
                        cov.fixedcov.append(word)
        
    def _numargs(self, arg):
        """Count number of arguments in a list (equivalent to SAS %numargs)"""
        if not arg:
            return 0
        if isinstance(arg, (list, tuple)):
            return len(arg)
        if isinstance(arg, str):
            # Split by spaces and count non-empty elements
            return len([x for x in arg.split() if x])
        return 1
        
    def run(self):
        """Main execution method for the g-formula"""
        
        # Process and validate parameters
        self._process_parameters()
        
        # Initialize bootstrap seeds
        self._initialize_seeds()
        
        # Process outcome categories for cateofu
        if self.outcome_type == 'cateofu':
            self._process_outcome_categories()
            
        # Process covariate dimensions and levels
        self._process_covariate_dimensions()
        
        # Prepare data
        self._dataprep()
        
        # Run main analysis loop
        results = self._run_analysis()
        
        return results
        
    def _initialize_seeds(self):
        """Initialize random seeds for bootstrap samples"""
        self.seed_list = [self.seed]
        self.seed_holder = self.seed
        
        for i in range(1, self.nsamples + 1):
            self.seed += 3
            self.seed_list.append(self.seed)
            
        # Set seed for current run
        self.seed = self.seed_list[self.sample_start]
        np.random.seed(self.seed)
        
        if self.printlogstats == 1:
            logger.info(f"seed = {self.seed}, seedlist = {self.seed_list}")
            
    def _process_outcome_categories(self):
        """Process outcome categories for categorical EOF outcomes"""
        if self.outcome_type == 'cateofu':
            # Get maximum level of outcome
            self.outclev = int(self.data[self.outcome].max())
            # Create knots for categories
            self.outcknots = list(range(1, self.outclev + 1))
            
            if self.printlogstats == 1:
                logger.info(f"Knots defining the categories for {self.outcome}: outcknots={self.outcknots}")
                
    def _process_covariate_dimensions(self):
        """Process covariate dimensions, levels, and arrays"""
        
        # Initialize use label flags
        if self.outcome_interact:
            self.uselabelo = 1
            
        # Process each covariate
        for i in range(0, self.ncov + 1):
            if i not in self.covariates:
                continue
                
            cov = self.covariates[i]
            
            # Process interaction terms
            if cov.interact:
                self.uselabelc = 1
                
            # Process categorical variables
            if cov.otype != 5 and ('cat' in cov.ptype or (cov.etype and 'cat' in cov.etype)):
                self._process_categorical_covariate(cov, i)
                
            # Process spline variables
            if 'spl' in cov.ptype:
                self._process_spline_covariate(cov, i)
                
            # Process tsswitch1 variables
            if cov.ptype == 'tsswitch1':
                self._process_tsswitch1_covariate(cov, i)
                
    def _process_categorical_covariate(self, cov, i):
        """Process categorical covariate levels and knots"""
        
        # Handle lag1cumavgcat and lag2cumavgcat
        if cov.ptype in ('lag1cumavgcat', 'lag2cumavgcat'):
            if isinstance(cov.knots, str) and ':' not in cov.knots:
                cov.cumavg_l1_knots = cov.knots
                cov.cumavg_l1_lev = self._numargs(cov.cumavg_l1_knots) + 1
            else:
                # Handle colon-separated knots
                knot_parts = cov.knots.split(':')
                cov.cumavg_l1_knots = knot_parts[1] if len(knot_parts) > 1 else cov.knots
                cov.cumavg_l1_lev = self._numargs(cov.cumavg_l1_knots) + 1
                cov.knots = knot_parts[0]
                
            if self.printlogstats == 1:
                logger.info(f"(knots, lev) for {cov.ptype} {cov.name}: ({cov.knots}, {self._numargs(cov.knots) + 1}) _cumavg ({cov.cumavg_l1_knots}, {cov.cumavg_l1_lev})")
                
        # Set levels for categorical ptypes
        if 'cat' in cov.ptype:
            cov.lev = self._numargs(cov.knots) + 1
            
        # Set levels for categorical etypes
        if cov.etype and 'cat' in cov.etype:
            cov.elev = self._numargs(cov.eknots) + 1
            
        if self.printlogstats == 1 and 'cat' in cov.ptype:
            logger.info(f"Number of categories of {cov.name}: cov{i}lev={cov.lev}")
            
    def _process_spline_covariate(self, cov, i):
        """Process spline covariate levels and knots"""
        
        if isinstance(cov.knots, list) and len(cov.knots) > 1:
            cov.lev = len(cov.knots)
        else:
            cov.lev = cov.knots if isinstance(cov.knots, (int, float)) else 0
            self.anyfindpknots = 1
            cov.findknots = 1
            
        if self.printlogstats == 1:
            logger.info(f"Number of levels of {cov.name}: cov{i}lev={cov.lev}")
            
    def _process_tsswitch1_covariate(self, cov, i):
        """Process tsswitch1 covariate levels"""
        
        if isinstance(cov.knots, list) and len(cov.knots) > 1:
            cov.lev = len(cov.knots)
        else:
            if cov.knots == 0:
                cov.lev = 1
            else:
                cov.lev = cov.knots if isinstance(cov.knots, (int, float)) else 0
                self.anyfindpknots = 1
                cov.findknots = 1
                
        if self.printlogstats == 1:
            logger.info(f"Number of categories of {cov.name}(tsswitch1): cov{i}lev={cov.lev}")
            
    def _dataprep(self):
        """Prepare data for analysis (equivalent to SAS %dataprep macro)"""
        
        # For EOF outcomes, filter data up to timepoints - 1
        if self.outcome_type in ('bineofu', 'conteofu', 'conteofu2', 'conteofu3', 'conteofu4', 'cateofu'):
            self.data = self.data[self.data[self.time] < self.time_points].copy()
            
        # Create interaction terms
        self._create_interactions()
        
        # Create lagged variables
        self._create_lagged_variables()
        
        # Create cumulative variables
        self._create_cumulative_variables()
        
        # Create spline variables
        if self.anyfindpknots or self.anyfindeknots:
            self._create_spline_knots()
            
    def _create_interactions(self):
        """Create interaction terms for all variables"""
        
        # Process outcome interactions
        if self.outcome_interact:
            self._create_variable_interactions(self.outcome, self.outcome_interact)
            
        # Process competing event interactions
        if self.compevent and self.compevent_interact:
            self._create_variable_interactions(self.compevent, self.compevent_interact)
            
        # Process censoring interactions
        if self.censor and self.censor_interact:
            self._create_variable_interactions(self.censor, self.censor_interact)
            
        # Process covariate interactions
        for i, cov in self.covariates.items():
            if cov.interact and cov.otype != 0:
                self._create_variable_interactions(cov.name, cov.interact)
                
    def _create_variable_interactions(self, var_name, interactions):
        """Create interaction terms for a single variable"""
        
        if isinstance(interactions, str):
            interactions = interactions.split()
            
        for interaction in interactions:
            # Parse interaction term (e.g., "var1*var2")
            if '*' in interaction:
                parts = interaction.split('*')
                if len(parts) == 2:
                    var1, var2 = parts
                    # Create interaction column
                    self.data[f"{var_name}_I_{var1}_{var2}"] = self.data[var1] * self.data[var2]
                    
    def _create_lagged_variables(self):
        """Create lagged variables for covariates"""
        
        # Sort by id and time
        self.data = self.data.sort_values([self.id, self.time])
        
        # Create lag indicators
        self.data['L1'] = (self.data[self.time] >= 1).astype(int)
        self.data['L2'] = (self.data[self.time] >= 2).astype(int)
        self.data['L3'] = (self.data[self.time] >= 3).astype(int)
        self.data['L4'] = (self.data[self.time] >= 4).astype(int)
        
        # Create lagged values for each covariate
        for i, cov in self.covariates.items():
            if i == 0:  # Skip time variable
                continue
                
            # Create baseline value
            self.data[f"{cov.name}_b"] = self.data.groupby(self.id)[cov.name].transform('first')
            
            # Create lagged values based on ptype
            if 'lag' in cov.ptype:
                for lag in range(1, 5):  # Create up to 4 lags
                    self.data[f"{cov.name}_l{lag}"] = self.data.groupby(self.id)[cov.name].shift(lag)
                    
    def _create_cumulative_variables(self):
        """Create cumulative variables"""
        
        for i, cov in self.covariates.items():
            if 'cum' in cov.ptype:
                # Create cumulative sum
                self.data[f"{cov.name}_cumsum"] = self.data.groupby(self.id)[cov.name].cumsum()
                
                # Create cumulative average
                self.data[f"{cov.name}_cumavg"] = (
                    self.data.groupby(self.id)[cov.name].expanding().mean().reset_index(0, drop=True)
                )
                
    def _create_spline_knots(self):
        """Find and create spline knots for variables that need them"""
        
        for i, cov in self.covariates.items():
            if cov.findknots and isinstance(cov.knots, (int, float)) and cov.knots > 0:
                # Calculate percentile knots
                percentiles = np.linspace(0, 100, int(cov.knots) + 2)[1:-1]
                cov.knots = np.percentile(self.data[cov.name].dropna(), percentiles).tolist()
                
                if self.printlogstats == 1:
                    logger.info(f"Calculated knots for {cov.name}: {cov.knots}")
                    
    def _run_analysis(self):
        """Run the main g-formula analysis"""
        
        # Initialize results storage
        results = {
            'parameters': {},
            'simulated_data': {},
            'risk_estimates': {},
            'bootstrap_results': []
        }
        
        # Run analysis for each bootstrap sample
        for bsample in range(self.sample_start, self.sample_end + 1):
            
            # Set seed for this sample
            if bsample > 0:
                self.seed = self.seed_list[bsample]
                np.random.seed(self.seed)
                
            # Create bootstrap sample if needed
            if bsample > 0:
                sample_data = self._create_bootstrap_sample()
            else:
                sample_data = self.data.copy()
                
            # Estimate parameters
            params = self._estimate_parameters(sample_data, bsample)
            
            if bsample == 0:
                results['parameters'] = params
                
            # Run simulations for interventions
            if self.numint > 0:
                sim_results = self._run_interventions(sample_data, params, bsample)
                
                if bsample == 0:
                    results['simulated_data'] = sim_results['simulated_data']
                    results['risk_estimates'] = sim_results['risk_estimates']
                else:
                    results['bootstrap_results'].append(sim_results['risk_estimates'])
                    
        return results
        
    def _get_parameters_summary(self):
        """Get summary of all parameters"""
        return {
            'outcome': self.outcome,
            'outcome_type': self.outcome_type,
            'time': self.time,
            'id': self.id,
            'covariates': {i: cov.name for i, cov in self.covariates.items()},
            'nsamples': self.nsamples,
            'nsimul': self.nsimul
        }
        
    def _create_bootstrap_sample(self):
        """Create a bootstrap sample of the data"""
        # Sample subjects with replacement
        unique_ids = self.data[self.id].unique()
        n_subjects = len(unique_ids)
        sampled_ids = np.random.choice(unique_ids, size=n_subjects, replace=True)
        
        # Create new data with sampled subjects
        sampled_data = []
        for new_id, orig_id in enumerate(sampled_ids):
            subject_data = self.data[self.data[self.id] == orig_id].copy()
            subject_data['newid'] = new_id
            sampled_data.append(subject_data)
            
        bootstrap_data = pd.concat(sampled_data, ignore_index=True)
        return bootstrap_data
        
    def _estimate_parameters(self, data, bsample):
        """Estimate parameters for all models"""
        
        params = {}
        
        # Add weights if not present
        if self.weight:
            data['_weight_'] = data[self.weight]
        else:
            data['_weight_'] = 1.0
            
        # Estimate outcome model parameters
        if self.outcome_type == 'binsurv':
            params['outcome'] = self._fit_binary_survival_model(data, bsample)
        elif self.outcome_type == 'bineofu':
            params['outcome'] = self._fit_binary_eof_model(data, bsample)
        elif self.outcome_type == 'conteofu':
            params['outcome'] = self._fit_continuous_eof_model(data, bsample)
        elif self.outcome_type == 'cateofu':
            params['outcome'] = self._fit_categorical_eof_model(data, bsample)
            
        # Estimate competing event model if specified
        if self.compevent:
            params['compevent'] = self._fit_competing_event_model(data, bsample)
            
        # Estimate censoring model if specified
        if self.censor:
            params['censor'] = self._fit_censoring_model(data, bsample)
            
        # Estimate covariate models
        for i in range(0, self.ncov + 1):
            if i not in self.covariates:
                continue
                
            cov = self.covariates[i]
            params[f'cov{i}'] = self._fit_covariate_model(data, cov, i, bsample)
            
        return params
        
    def _fit_binary_survival_model(self, data, bsample):
        """Fit binary survival outcome model"""
        
        # Create predictor list
        predictors = self._get_outcome_predictors()
        
        # Filter data based on outcome where condition
        model_data = data.copy()
        if self.outcome_wherem != "(1=1)":
            model_data = model_data.query(self.outcome_wherem)
            
        # Fit logistic regression
        try:
            formula = f"{self.outcome} ~ " + " + ".join(predictors)
            model = sm.GLM.from_formula(
                formula,
                data=model_data,
                family=sm.families.Binomial(),
                freq_weights=model_data['_weight_']
            ).fit()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit outcome model for sample {bsample}: {e}")
            return None
            
    def _fit_binary_eof_model(self, data, bsample):
        """Fit binary end-of-follow-up outcome model"""
        
        # Filter to last time point
        model_data = data[data[self.time] == self.time_points - 1].copy()
        
        # Create predictor list
        predictors = self._get_outcome_predictors()
        
        # Apply where condition
        if self.outcome_wherem != "(1=1)":
            model_data = model_data.query(self.outcome_wherem)
            
        # Fit logistic regression
        try:
            formula = f"{self.outcome} ~ " + " + ".join(predictors)
            model = sm.GLM.from_formula(
                formula,
                data=model_data,
                family=sm.families.Binomial(),
                freq_weights=model_data['_weight_']
            ).fit()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit EOF outcome model for sample {bsample}: {e}")
            return None
            
    def _fit_continuous_eof_model(self, data, bsample):
        """Fit continuous end-of-follow-up outcome model"""
        
        # Filter to last time point
        model_data = data[data[self.time] == self.time_points - 1].copy()
        
        # Create predictor list
        predictors = self._get_outcome_predictors()
        
        # Apply where condition
        if self.outcome_wherem != "(1=1)":
            model_data = model_data.query(self.outcome_wherem)
            
        # Remove missing outcomes
        model_data = model_data[model_data[self.outcome].notna()]
        
        # Fit linear regression
        try:
            formula = f"{self.outcome} ~ " + " + ".join(predictors)
            model = sm.WLS.from_formula(
                formula,
                data=model_data,
                weights=model_data['_weight_']
            ).fit()
            
            # Store min/max for truncation if needed
            outcome_min = model_data[self.outcome].min()
            outcome_max = model_data[self.outcome].max()
            outcome_std = model.resid.std()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'std': outcome_std,
                'min': outcome_min,
                'max': outcome_max,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit continuous EOF model for sample {bsample}: {e}")
            return None
            
    def _fit_categorical_eof_model(self, data, bsample):
        """Fit categorical end-of-follow-up outcome model"""
        
        # Filter to last time point
        model_data = data[data[self.time] == self.time_points - 1].copy()
        
        # Create predictor list
        predictors = self._get_outcome_predictors()
        
        # Apply where condition
        if self.outcome_wherem != "(1=1)":
            model_data = model_data.query(self.outcome_wherem)
            
        # Fit multinomial/ordinal logistic regression
        # This is a simplified version - actual implementation would use
        # appropriate multinomial/ordinal regression
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = model_data[predictors]
            y = model_data[self.outcome]
            weights = model_data['_weight_']
            
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            model.fit(X, y, sample_weight=weights)
            
            return {
                'model': model,
                'classes': model.classes_,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit categorical EOF model for sample {bsample}: {e}")
            return None
            
    def _fit_competing_event_model(self, data, bsample):
        """Fit competing event model"""
        
        # Similar to binary survival but for competing event
        predictors = self._get_compevent_predictors()
        
        model_data = data.copy()
        if self.compevent_wherem != "(1=1)":
            model_data = model_data.query(self.compevent_wherem)
            
        try:
            formula = f"{self.compevent} ~ " + " + ".join(predictors)
            model = sm.GLM.from_formula(
                formula,
                data=model_data,
                family=sm.families.Binomial(),
                freq_weights=model_data['_weight_']
            ).fit()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit competing event model for sample {bsample}: {e}")
            return None
            
    def _fit_censoring_model(self, data, bsample):
        """Fit censoring model"""
        
        predictors = self._get_censor_predictors()
        
        model_data = data.copy()
        if self.censor_wherem != "(1=1)":
            model_data = model_data.query(self.censor_wherem)
            
        try:
            formula = f"{self.censor} ~ " + " + ".join(predictors)
            model = sm.GLM.from_formula(
                formula,
                data=model_data,
                family=sm.families.Binomial(),
                freq_weights=model_data['_weight_']
            ).fit()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit censoring model for sample {bsample}: {e}")
            return None
            
    def _fit_covariate_model(self, data, cov, i, bsample):
        """Fit model for a single covariate"""
        
        # Skip if covariate has no model (otype=0)
        if cov.otype == 0:
            return None
            
        # Get predictors for this covariate
        predictors = self._get_covariate_predictors(cov, i)
        
        # Apply where conditions
        model_data = data[data[self.time] > 0].copy()
        
        # Apply skip times
        if cov.skip and cov.skip != -1:
            skip_times = cov.skip if isinstance(cov.skip, list) else [cov.skip]
            model_data = model_data[~model_data[self.time].isin(skip_times)]
            
        # Apply where condition
        if cov.wherem != "(1=1)":
            model_data = model_data.query(cov.wherem)
            
        # Apply visit process if specified
        if hasattr(cov, 'usevisitp') and cov.usevisitp == 1:
            model_data = model_data[model_data[cov.randomvisitp] == 1]
            
        # Fit appropriate model based on otype
        if cov.otype == 1:  # Binary
            return self._fit_binary_covariate(model_data, cov, predictors, bsample)
        elif cov.otype == 2:  # Zero-inflated binary
            return self._fit_zero_inflated_binary(model_data, cov, predictors, bsample)
        elif cov.otype == 3:  # Continuous
            return self._fit_continuous_covariate(model_data, cov, predictors, bsample)
        elif cov.otype == 4:  # Categorical
            return self._fit_categorical_covariate(model_data, cov, predictors, bsample)
        elif cov.otype == 5:  # Bounded continuous
            return self._fit_bounded_continuous(model_data, cov, predictors, bsample)
        elif cov.otype == 6:  # Zero-inflated normal
            return self._fit_zero_inflated_normal(model_data, cov, predictors, bsample)
        elif cov.otype == 7:  # Truncated normal
            return self._fit_truncated_normal(model_data, cov, predictors, bsample)
            
    def _fit_binary_covariate(self, data, cov, predictors, bsample):
        """Fit binary covariate model"""
        try:
            formula = f"{cov.name} ~ " + " + ".join(predictors)
            model = sm.GLM.from_formula(
                formula,
                data=data,
                family=sm.families.Binomial(),
                freq_weights=data['_weight_']
            ).fit()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit model for {cov.name} sample {bsample}: {e}")
            return None
            
    def _fit_continuous_covariate(self, data, cov, predictors, bsample):
        """Fit continuous covariate model"""
        try:
            formula = f"{cov.name} ~ " + " + ".join(predictors)
            model = sm.WLS.from_formula(
                formula,
                data=data,
                weights=data['_weight_']
            ).fit()
            
            # Store min/max for truncation
            var_min = data[cov.name].min()
            var_max = data[cov.name].max()
            var_std = model.resid.std()
            
            return {
                'coefficients': model.params.to_dict(),
                'vcov': model.cov_params().values,
                'std': var_std,
                'min': var_min,
                'max': var_max,
                'sample': bsample
            }
        except Exception as e:
            logger.warning(f"Failed to fit model for {cov.name} sample {bsample}: {e}")
            return None
            
    def _get_outcome_predictors(self):
        """Get predictor list for outcome model"""
        predictors = []
        
        # Add fixed covariates
        if self.fixedcov:
            predictors.extend(self.fixedcov)
            
        # Add time-varying covariates
        for i in range(1, self.ncov + 1):
            if i in self.covariates:
                cov = self.covariates[i]
                if cov.mtype.upper() in ('ALL', 'SOME'):
                    predictors.append(cov.name)
                    
        # Add additional variables
        if self.eventaddvars:
            addvars = self.eventaddvars if isinstance(self.eventaddvars, list) else self.eventaddvars.split()
            predictors.extend(addvars)
            
        # Add interaction terms
        if self.outcome_interact:
            # Process interaction terms
            pass
            
        return predictors
        
    def _get_covariate_predictors(self, cov, i):
        """Get predictor list for a covariate model"""
        predictors = []
        
        # Add fixed covariates specific to this covariate
        if hasattr(cov, 'fixedcov') and cov.fixedcov:
            predictors.extend(cov.fixedcov)
            
        # Add lagged values and other predictors based on ptype
        # This is a simplified version - actual implementation would
        # handle all the different ptypes properly
        
        # Add additional variables
        if cov.addvars:
            addvars = cov.addvars if isinstance(cov.addvars, list) else cov.addvars.split()
            predictors.extend(addvars)
            
        return predictors
        
    def _get_compevent_predictors(self):
        """Get predictor list for competing event model"""
        # Similar to outcome predictors but with compeventaddvars
        predictors = self._get_outcome_predictors()
        
        if self.compeventaddvars:
            addvars = self.compeventaddvars if isinstance(self.compeventaddvars, list) else self.compeventaddvars.split()
            predictors.extend(addvars)
            
        return predictors
        
    def _get_censor_predictors(self):
        """Get predictor list for censoring model"""
        predictors = self._get_outcome_predictors()
        
        if self.censoraddvars:
            addvars = self.censoraddvars if isinstance(self.censoraddvars, list) else self.censoraddvars.split()
            predictors.extend(addvars)
            
        return predictors
        
    def _fit_zero_inflated_binary(self, data, cov, predictors, bsample):
        """Fit zero-inflated binary model (otype=2)"""
        # Filter to where lagged value is 0
        model_data = data[data[f"{cov.name}_l1"] == 0].copy()
        
        return self._fit_binary_covariate(model_data, cov, predictors, bsample)
        
    def _fit_categorical_covariate(self, data, cov, predictors, bsample):
        """Fit categorical covariate model (otype=4)"""
        # This would implement multinomial/ordinal logistic regression
        # Simplified version here
        return self._fit_binary_covariate(data, cov, predictors, bsample)
        
    def _fit_bounded_continuous(self, data, cov, predictors, bsample):
        """Fit bounded continuous model (otype=5)"""
        # This would implement beta regression or similar
        # Simplified version here
        return self._fit_continuous_covariate(data, cov, predictors, bsample)
        
    def _fit_zero_inflated_normal(self, data, cov, predictors, bsample):
        """Fit zero-inflated normal model (otype=6)"""
        # This would implement a two-part model
        # Simplified version here
        return self._fit_continuous_covariate(data, cov, predictors, bsample)
        
    def _fit_truncated_normal(self, data, cov, predictors, bsample):
        """Fit truncated normal model (otype=7)"""
        # This would implement truncated regression
        # Simplified version here
        return self._fit_continuous_covariate(data, cov, predictors, bsample)
        
    def _run_interventions(self, data, params, bsample):
        """Run intervention simulations"""
        # This is a placeholder for intervention simulation logic
        # Would need to implement the full simulation algorithm
        
        sim_results = {
            'simulated_data': {},
            'risk_estimates': {}
        }
        
        return sim_results