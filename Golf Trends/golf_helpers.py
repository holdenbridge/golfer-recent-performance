import json
import os
import pandas as pd # type: ignore
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import linregress
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score)

def load_event_odds(event_name, folder_path):
    """Load odds JSON for an event and return a dict mapping PlayerName -> {WinOdds, T5Odds, T10Odds, T20Odds}."""
    odds_path = f'{folder_path}/{event_name}_odds.json'
    if not os.path.exists(odds_path):
        return {}
    with open(odds_path, 'r') as f:
        odds_data = json.load(f)

    def reformat(name):
        if isinstance(name, str) and ',' in name:
            last, first = [x.strip() for x in name.split(',', 1)]
            return f"{first} {last}"
        return name

    player_odds = {}
    for market_key, col_name in [('win_odds', 'WinOdds'), ('T5_odds', 'T5Odds'),
                                  ('T10_odds', 'T10Odds'), ('T20_odds', 'T20Odds')]:
        for entry in odds_data.get(market_key, []):
            pname = reformat(entry['player_name'])
            if pname not in player_odds:
                player_odds[pname] = {}
            player_odds[pname][col_name] = entry.get('close_odds', None)
    return player_odds

def evaluate_event(event_start_date, event_end_date, owgr_dict, event_name, folder_path, tournament_fields):
    weeks_before = range(1, 13)
    base_date = datetime.strptime(event_start_date, "%Y-%m-%d")
    week_dates = {}
    for w in weeks_before:
        target_date = base_date - timedelta(weeks=w)
        week_dates[f'Avg_Points_{w}weekbefore'] = target_date.strftime("%Y-%m-%d")

    player_odds = load_event_odds(event_name, folder_path)

    rows = []
    for player, data in owgr_dict.items():
        if player not in tournament_fields[event_name]:
            continue
        row = {'PlayerName': player}
        for colname, week_date in week_dates.items():
            row[colname] = data.get(week_date, {}).get('Avg_Points', None)
        row['Avg_Points_StartEvent'] = data.get(event_start_date, {}).get('Avg_Points', None)
        row['Avg_Points_EndEvent'] = data.get(event_end_date, {}).get('Avg_Points', None)
        if row['Avg_Points_StartEvent'] is not None:
            row['EventName'] = event_name
            row['FinishingPosition'] = tournament_fields[event_name][player]['Finishing Position']
            odds = player_odds.get(player, {})
            row['WinOdds'] = odds.get('WinOdds', None)
            row['T5Odds'] = odds.get('T5Odds', None)
            row['T10Odds'] = odds.get('T10Odds', None)
            row['T20Odds'] = odds.get('T20Odds', None)
            rows.append(row)

    return pd.DataFrame(rows)

def prepare_modeling_df(df):
    df_result = df.copy()
    df_result['EventChange'] = df_result['Avg_Points_EndEvent'] - df_result['Avg_Points_StartEvent']
    for i in range(1, 13):
        week_col = f'Avg_Points_{i}weekbefore'
        df_result[f'Week{i}Change'] = df_result['Avg_Points_StartEvent'] - df_result.get(week_col)
    # Also include Avg_Points_StartEvent and EventName in the resulting DataFrame
    df_result = df_result[[col for col in df_result.columns if ('Change' in col or col == 'PlayerName') or ('Odds' in col or col == 'PlayerName') or col == 'Avg_Points_StartEvent' or col == 'EventName' or col == 'FinishingPosition']]
    return df_result

def american_odds_to_implied_prob(american_odds):
    american_odds = float(str(american_odds).replace('+', ''))
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

def estimate_finishing_position(row):
    t20_prob = row['T20Prob']
    
    if t20_prob > 0.5:
        return 10
    elif t20_prob > 0.4:
        return 20
    elif t20_prob > 0.2:
        return 40
    else:
        return 65

def calc_slope(row):
    weeks = [row[f'Week{i}Change'] for i in range(1, 13)]
    slope, _, _, _, _ = linregress(range(12), weeks)
    return slope

def recent_streak(row):
    streak = 0
    for i in range(1, 13):
        if row[f'Week{i}Change'] > 0:
            streak += 1
        else:
            break
    return streak


class BinaryXGBClassifier:
    def __init__(
        self,
        X,
        y,
        param_grid,
        use_gpu=False,
        loss_func="accuracy",
        cv=5,
        random_state=33,
        eval_metric="auc",
    ):
        from sklearn.model_selection import train_test_split

        self.loss_func = loss_func
        self.cv = cv
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.tree_method = "gpu_hist" if use_gpu else "hist"
        self.param_grid = param_grid

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

    def fit(self):
        if self.eval_metric == "f1":
            model = XGBClassifier(
                random_state=self.random_state,
                eval_metric="auc",
                objective="binary:logistic",
                tree_method=self.tree_method,
            )
        else:
            model = XGBClassifier(
                random_state=self.random_state,
                eval_metric=self.eval_metric,
                objective="binary:logistic",
                tree_method=self.tree_method,
            )

        self.grid = GridSearchCV(
            model,
            self.param_grid,
            scoring=self.loss_func,
            cv=self.cv,
            verbose=1,
            n_jobs=-1,
        )
        self.grid.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.grid.predict(X_test)

    def predict_probs(self, X_test):
        return self.grid.predict_proba(X_test)

    def score(self, X_test, y_test):
        threshold = 0.5
        y_pred = self.predict_probs(X_test)
        y_pred = y_pred[:, 1]
        round_y_pred = (y_pred >= threshold).astype(int)
        acc = accuracy_score(y_test, round_y_pred)
        f1 = f1_score(y_test, round_y_pred)
        log_loss_score = log_loss(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, round_y_pred)
        recall = recall_score(y_test, round_y_pred)
        balanced_acc = balanced_accuracy_score(y_test, round_y_pred)

        # print each metric with the title of the metric in all caps, and the score rounded to 2 decimal places
        print("THRESHOLD:", round(threshold, 2))
        print("ACCURACY:", round(acc, 2))
        print("F1 SCORE:", round(f1, 2))
        print("LOG LOSS:", round(log_loss_score, 2))
        print("ROC AUC:", round(roc_auc, 2))
        print("PRECISION:", round(precision, 2))
        print("RECALL:", round(recall, 2))
        print("BALANCED ACCURACY:", round(balanced_acc, 2))

        # return labeled dictionary of scores
        return {
            "accuracy": acc,
            "f1": f1,
            "log_loss": log_loss_score,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "balanced_accuracy": balanced_acc,
        }

    def plot_roc_auc_curve(self, X_test, y_test):
        from matplotlib import pyplot as plt
        from sklearn.metrics import roc_curve

        y_pred = self.predict_probs(X_test)
        y_pred = y_pred[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, color="orange", label="ROC")
        plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()

    def plot_feature_importances(self, title="Feature Importances"):
        import matplotlib.pyplot as plt
        from numpy import argsort

        # plot feature importances from most important to least important
        # and the x labels are the names of the features
        plt.figure(figsize=(15, 15))
        plt.title(title)
        features = self.X_train.columns
        importances = self.grid.best_estimator_.feature_importances_
        indices = argsort(importances)
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()

    def run(self):
        # this function should fit the model and return a tuple of (fitted final model, score, best params)
        self.fit()
        self.plot_feature_importances()
        self.plot_roc_auc_curve(self.X_test, self.y_test)
        scores = self.score(self.X_test, self.y_test)
        return self.grid.best_estimator_, scores, self.grid.best_params_