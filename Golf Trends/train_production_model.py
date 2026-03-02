import json
import pandas as pd
import warnings
import pickle
import datetime
from golf_helpers import evaluate_event, prepare_modeling_df, american_odds_to_implied_prob, estimate_finishing_position, calc_slope, recent_streak, BinaryXGBClassifier
warnings.filterwarnings('ignore')

folder_path = '/Users/holden.bridge/Desktop/golf-research/Golf Trends/owgr_historical'
with open(f'{folder_path}/owgr_dict.json', 'r') as f:
        owgr_dict = json.load(f)
with open(f'{folder_path}/tournament_fields.json', 'r') as f:
        tournament_fields = json.load(f)

# 2025 Season
df_players2025 = evaluate_event("2025-03-08", "2025-03-15", owgr_dict, "PlayersChampionship2025", folder_path, tournament_fields)
df_masters2025 = evaluate_event("2025-04-05", "2025-04-12", owgr_dict, "Masters2025", folder_path, tournament_fields)
df_PGA_Championship2025 = evaluate_event("2025-05-10", "2025-05-17", owgr_dict, "PGAChampionship2025", folder_path, tournament_fields)
df_US_Open2025 = evaluate_event("2025-06-07", "2025-06-14", owgr_dict, "USOpen2025", folder_path, tournament_fields)
df_Open_Championship2025 = evaluate_event("2025-07-12", "2025-07-19", owgr_dict, "OpenChampionship2025", folder_path, tournament_fields)

# 2024 Season
df_players2024 = evaluate_event("2024-03-09", "2024-03-16", owgr_dict, "PlayersChampionship2024", folder_path, tournament_fields)
df_masters2024 = evaluate_event("2024-04-06", "2024-04-13", owgr_dict, "Masters2024", folder_path, tournament_fields)
df_PGA_Championship2024 = evaluate_event("2024-05-11", "2024-05-18", owgr_dict, "PGAChampionship2024", folder_path, tournament_fields)
df_US_Open2024 = evaluate_event("2024-06-08", "2024-06-15", owgr_dict, "USOpen2024", folder_path, tournament_fields)
df_Open_Championship2024 = evaluate_event("2024-07-13", "2024-07-20", owgr_dict, "OpenChampionship2024", folder_path, tournament_fields)

# 2023 Season
df_players2023 = evaluate_event("2023-03-04", "2023-03-11", owgr_dict, "PlayersChampionship2023", folder_path, tournament_fields)
df_masters2023 = evaluate_event("2023-04-01", "2023-04-08", owgr_dict, "Masters2023", folder_path, tournament_fields)
df_PGA_Championship2023 = evaluate_event("2023-05-13", "2023-05-20", owgr_dict, "PGAChampionship2023", folder_path, tournament_fields)
df_US_Open2023 = evaluate_event("2023-06-10", "2023-06-17", owgr_dict, "USOpen2023", folder_path, tournament_fields)
df_Open_Championship2023 = evaluate_event("2023-07-15", "2023-07-22", owgr_dict, "OpenChampionship2023", folder_path, tournament_fields)

# Prepare 2025 for modeling
df_modeling_players2025 = prepare_modeling_df(df_players2025)
df_modeling_masters2025 = prepare_modeling_df(df_masters2025)
df_modeling_pga_championship2025 = prepare_modeling_df(df_PGA_Championship2025)
df_modeling_us_open2025 = prepare_modeling_df(df_US_Open2025)
df_modeling_open_championship2025 = prepare_modeling_df(df_Open_Championship2025)

# Prepare 2024 for modeling
df_modeling_players2024 = prepare_modeling_df(df_players2024)
df_modeling_masters2024 = prepare_modeling_df(df_masters2024)
df_modeling_pga_championship2024 = prepare_modeling_df(df_PGA_Championship2024)
df_modeling_us_open2024 = prepare_modeling_df(df_US_Open2024)
df_modeling_open_championship2024 = prepare_modeling_df(df_Open_Championship2024)

# Prepare 2023 for modeling
df_modeling_players2023 = prepare_modeling_df(df_players2023)
df_modeling_masters2023 = prepare_modeling_df(df_masters2023)
df_modeling_pga_championship2023 = prepare_modeling_df(df_PGA_Championship2023)
df_modeling_us_open2023 = prepare_modeling_df(df_US_Open2023)
df_modeling_open_championship2023 = prepare_modeling_df(df_Open_Championship2023)

# Create Final Modeling DF
df_modeling = pd.concat([
df_modeling_players2023, df_modeling_masters2023, df_modeling_pga_championship2023, df_modeling_us_open2023, df_modeling_open_championship2023,
df_modeling_players2024, df_modeling_masters2024, df_modeling_pga_championship2024, df_modeling_us_open2024, df_modeling_open_championship2024,
df_modeling_players2025, df_modeling_masters2025, df_modeling_pga_championship2025, df_modeling_us_open2025, df_modeling_open_championship2025,
])

# Feature Engineering
feature_cols = [f'Week{i}Change' for i in [4, 8, 12]]
drop_cols = feature_cols + ['EventChange', 'WinOdds', 'T5Odds', 'T10Odds', 'T20Odds']
df_model_clean = df_modeling.dropna()
df_model_clean['T20Prob'] = df_model_clean['T20Odds'].apply(american_odds_to_implied_prob)
df_model_clean['EstimatedFinishingPosition'] = df_model_clean.apply(estimate_finishing_position, axis=1)
df_model_clean['BeatExpectation'] = (df_model_clean['FinishingPosition'] < df_model_clean['EstimatedFinishingPosition']).astype(int)
df_model_clean['momentum_slope'] = df_model_clean.apply(calc_slope, axis=1)
df_model_clean['recent_momentum'] = df_model_clean[['Week1Change', 'Week2Change', 'Week3Change', 'Week4Change']].mean(axis=1)
df_model_clean['distant_momentum'] = df_model_clean[['Week9Change', 'Week10Change', 'Week11Change', 'Week12Change']].mean(axis=1)
df_model_clean['recent_vs_distant'] = df_model_clean['recent_momentum'] - df_model_clean['distant_momentum']
df_model_clean['momentum_volatility'] = df_model_clean[[f'Week{i}Change' for i in range(1, 13)]].std(axis=1)
df_model_clean['best_week'] = df_model_clean[[f'Week{i}Change' for i in range(1, 13)]].max(axis=1)
df_model_clean['worst_week'] = df_model_clean[[f'Week{i}Change' for i in range(1, 13)]].min(axis=1)
df_model_clean['week_range'] = df_model_clean['best_week'] - df_model_clean['worst_week']
df_model_clean['positive_week_count'] = (df_model_clean[[f'Week{i}Change' for i in range(1, 13)]] > 0).sum(axis=1)
df_model_clean['recent_positive_streak'] = df_model_clean.apply(recent_streak, axis=1)
df_model_clean['acceleration_4v12'] = df_model_clean['Week4Change'] - df_model_clean['Week12Change']
df_model_clean['points_x_momentum8'] = df_model_clean['Avg_Points_StartEvent'] * df_model_clean['Week8Change']

# Create predictors and target variable
X = df_model_clean[['Avg_Points_StartEvent', 'Week4Change', 'recent_vs_distant', 'recent_positive_streak', 'acceleration_4v12', 'points_x_momentum8']]
y = df_model_clean['BeatExpectation']

# Calculate scale_pos_weight ratio
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_weight = neg_count / pos_count
print(f"scale_pos_weight: {scale_weight:.2f} ({neg_count} neg / {pos_count} pos)")

# Train Classifier (Beat Expectation)
grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [75, 100, 175, 250],
    'min_child_weight': [10],
    'gamma': [1, 2],
    'scale_pos_weight': [scale_weight]
}

model = BinaryXGBClassifier(X=X, y=y, param_grid=grid, cv=5).run()
xgb_model = model[0]

with open(f'beat_expectation_model_{datetime.datetime.now().strftime("%m_%d_26")}.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)