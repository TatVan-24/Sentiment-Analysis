from xgboost import XGBClassifier

def xg_boost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)