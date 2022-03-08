PARAMS = {
    "eta": 0.3,
    "max_depth": 3,
    "objective": "binary:logistic"
}

NUM_BOOST_ROUND = 300

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
