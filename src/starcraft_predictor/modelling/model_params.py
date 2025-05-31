UNIQUE_ID = "filehash"

PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "reg_alpha": 1,
    "min_child_weight": 50,
    "n_estimators": 100,
    "random_state": 2709,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
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

TARGET = "winner"
