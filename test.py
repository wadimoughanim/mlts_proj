from pathlib import Path
from typing import Dict, Any

import numpy as np
from pathlib import Path

from timeeval import TimeEval, MultiDatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters
from timeeval.resource_constraints import GB
from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
# from timeeval.algorithms import subsequence_if
from TimeEval.timeeval.algorithms import subsequence_if
from timeeval.params import FixedParameters
# /Users/wadimoughanim/Documents/education/Master_MVA/mlts_proj/TimeEval/timeeval/algorithms/subsequence_if.py
#
path = '/Users/wadimoughanim/Documents/education/Master_MVA/mlts_proj/timeeval-datasets'
dm = DatasetManager(Path(path))
datasets = []
# datasets.append(dm.select(collection="CalIt2"))
# datasets.append(dm.select(collection="Daphnet"))
datasets = dm.select(collection="CalIt2")  #+ dm.select(collection="Daphnet")
# Select datasets and algorithms
# Add algorithms to evaluate...
algorithms = [Algorithm(
    name="HIF",
    main=DockerAdapter(
        image_name="ghcr.io/timeeval/lof",
        tag="0.3.0",  # please use a specific tag instead of "latest" for reproducibility
        skip_pull=True  # set to True because the image is already present from the previous section
    ),
    # The hyperparameters of the algorithm are specified here. If you want to perform a parameter
    # search, you can also perform simple grid search with TimeEval using FullParameterGrid or
    # IndependentParameterGrid.
    param_config=FixedParameters({
        "n_neighbors": 50,
        "random_state": 42
    }),
    # required by DockerAdapter
    data_as_file=True,
    # You must specify the algorithm metadata here. The categories for all TimeEval algorithms can
    # be found in their README or their manifest.json-File.
    # UNSUPERVISED --> no training, SEMI_SUPERVISED --> training on normal data, SUPERVISED --> training on anomalies
    # if SEMI_SUPERVISED or SUPERVISED, the datasets must have a corresponding training time series
    training_type=TrainingType.UNSUPERVISED,
    # MULTIVARIATE (multidimensional TS) or UNIVARIATE (just a single dimension is supported)
    input_dimensionality=InputDimensionality.MULTIVARIATE
)]
timeeval = TimeEval(dm, datasets, algorithms)

# execute evaluation
timeeval.run()
# retrieve results
print(timeeval.get_results())