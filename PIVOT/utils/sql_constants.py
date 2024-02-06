from collections import OrderedDict
from typing import Dict, Any, Optional

# Matching argument types for stored procedures
SP_ARGS_TYPE_MAPPING: Dict[str, Optional[OrderedDict[str, Any]]] = {
    "AL_RANKINGS": OrderedDict([
        ('MODEL_ID', int),
        ('D_METRIC_ID', int),
        ('RELABEL_LAMBDA', float),
        ('BATCH_SIZE', int)
    ]),
    "MODEL_EVALUATION_NON_TEST": OrderedDict([
        ('MODEL_ID', int),
        ('MINIMUM_PERCENT', float)
    ]),
    "MODEL_EVALUATION_MAX_CONSENSUS_FILTERING": OrderedDict([
        ('MODEL_ID', int),
        ('MINIMUM_PERCENT', float)
    ]),
    "AL_TRAIN_SET": OrderedDict([
        ('MODEL_ID', int),
        ('D_METRIC_ID', int),
        ('TRAIN_SIZE', int),
        ('IMAGE_IDS', str)

    ]),
    "GENERATE_RANDOM_TEST_SET": OrderedDict([
        ('TEST_SIZE', int),
        ('IMAGE_IDS', str)
    ]),
    "GENERATE_IMAGES_TO_PREDICT": OrderedDict([
        ('MODEL_ID', int)
    ]),
    "GENERATE_IMAGES_TO_METRIZE": OrderedDict([
        ('MODEL_ID', int),
        ('D_METRIC_ID', int),
    ])
}

# File names for stored procedures
SP_FILE_NAMES: Dict[str, str] = {
    "AL_RANKINGS": "./stored_procedures/Labeling_Ranking.sql",
    "MODEL_EVALUATION_NON_TEST": "./stored_procedures/Model_Evaluation_NonTest.sql",
    "MODEL_EVALUATION_MAX_CONSENSUS_FILTERING": "./stored_procedures/Model_Evaluation_Filtering.sql",
    "AL_TRAIN_SET": "./stored_procedures/Model_Training.sql",
    "GENERATE_RANDOM_TEST_SET": "./stored_procedures/Generate_Random_Test_Set.sql",
    "GENERATE_IMAGES_TO_PREDICT": "./stored_procedures/Images_To_Predict.sql",
    "GENERATE_IMAGES_TO_METRIZE": "./stored_procedures/Images_To_Metrize.sql"
}

