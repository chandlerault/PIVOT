from collections import OrderedDict
from typing import Dict, Any, Optional

# Matching argument types for stored procedures
SP_ARGS_TYPE_MAPPING: Dict[str, Optional[OrderedDict[str, Any]]] = {
    "AL_RANKINGS": OrderedDict([
        ('M_ID', int),
        ('D_ID', int),
        ('RELABEL_LAMBDA', float),
        ('BATCH_SIZE', int)
    ]),
    "MODEL_EVALUATION_MAX_CONSENSUS": OrderedDict([
        ('MODEL_ID', int)
    ]),
    "MODEL_EVALUATION_MAX_CONSENSUS_FILTERING": OrderedDict([
        ('MODEL_ID', int),
        ('MINIMUM_PERCENT', float)
    ]),
    "AL_TRAIN_SET": OrderedDict([
        ('MODEL_ID', int),
        ('D_METRIC_ID', int),
        ('TRAIN_SIZE', int)
    ]),
}

# File names for stored procedures
SP_FILE_NAMES: Dict[str, str] = {
    "AL_RANKINGS": "./stored_procedures/Labeling_Ranking.sql",
    "MODEL_EVALUATION_MAX_CONSENSUS": "./stored_procedures/Model_Evaluation.sql",
    "MODEL_EVALUATION_MAX_CONSENSUS_FILTERING": "./stored_procedures/Model_Evaluation_Filtering.sql",
    "AL_TRAIN_SET": "./stored_procedures/Model_Training.sql"
}
