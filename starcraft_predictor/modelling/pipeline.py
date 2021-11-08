from sklearn.pipeline import Pipeline

import starcraft_predictor.modelling.custom_transformers as ct


DIFF_COLUMNS = [
    (
        "food_made_1",
        "food_made_2",
    ),
    (
        "food_used_1",
        "food_used_2",
    ),
    (
        "minerals_collection_rate_1",
        "minerals_collection_rate_2",
    ),
    (
        "minerals_lost_1",
        "minerals_lost_2",
    ),
    (
        "minerals_lost_army_1",
        "minerals_lost_army_2",
    ),
    (
        "minerals_used_current_1",
        "minerals_used_current_2",
    ),
    (
        "minerals_used_current_army_1",
        "minerals_used_current_army_2",
    ),
    (
        "minerals_used_current_economy_1",
        "minerals_used_current_economy_2",
    ),
    (
        "vespene_collection_rate_1",
        "vespene_collection_rate_2",
    ),
    (
        "vespene_lost_1",
        "vespene_lost_2",
    ),
    (
        "vespene_lost_army_1",
        "vespene_lost_army_2",
    ),
    (
        "vespene_used_current_1",
        "vespene_used_current_2",
    ),
    (
        "vespene_used_current_army_1",
        "vespene_used_current_army_2",
    ),
    (
        "vespene_used_current_economy_1",
        "vespene_used_current_economy_2",
    ),
    (
        "vespene_used_current_technology_1",
        "vespene_used_current_technology_2",
    ),
    (
        "workers_active_count_1",
        "workers_active_count_2",
    ),
]

NEW_COLUMNS = [
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

# define the pre-processing pipeline used for prepping data
# for modelling
sc2_preprocessing_pipeline = Pipeline([
    (
        "column_difference_transformer",
        ct.ColumnDifferenceTransformer(
            diff_columns=DIFF_COLUMNS,
            new_columns=NEW_COLUMNS,
            drop_cols=True,
        ),
    ),
    (
        "passthrough", "passthrough",
    ),
])
