import numpy as np


PARAMS = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'colsample_bytree': 0.9,
    'enable_categorical': False,
    'gamma': 0,
    'gpu_id': -1,
    'importance_type': None,
    'interaction_constraints': '',
    'learning_rate': 0.1,
    'max_delta_step': 0,
    'max_depth': 4,
    'min_child_weight': 1,
    'missing': np.nan,
    'monotone_constraints': '()',
    'n_estimators': 500,
    'n_jobs': 8,
    'num_parallel_tree': 1,
    'predictor': 'auto',
    'random_state': 2709,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'subsample': 0.75,
    'tree_method': 'exact',
    'validate_parameters': 1,
    'verbosity': None
}

FEATURES = [
    "seconds",
    "food_made_diff",
    "food_used_diff",
    "minerals_collection_rate_diff",
    "minerals_lost_diff",
    "minerals_lost_army_diff",
    "minerals_used_current_diff",
    "minerals_used_current_army_diff",
    "minerals_used_current_economy_diff",
    "vespene_collection_rate_diff",
    "vespene_lost_diff",
    "vespene_lost_army_diff",
    "vespene_used_current_diff",
    "vespene_used_current_army_diff",
    "vespene_used_current_economy_diff",
    "vespene_used_current_technology_diff",
    "workers_active_count_diff",
]

RESPONSE = "winner"
