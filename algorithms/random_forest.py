from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def process_data(X_train, X_test, y_train, y_test, data):
    try:
        # Random Forest 모델 학습
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)

        # 특성 중요도 계산 및 선택
        importances = rf.feature_importances_
        feature_names = data.columns[:-1]
        feature_importance_dict = dict(zip(feature_names, importances))
        important_features_mask = importances > 0.02

        # 중요 특성만 선택하여 데이터 재구성
        X_train = X_train[:, important_features_mask]
        X_test = X_test[:, important_features_mask]

        # 교차 검증
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        accuracy_percentage = round(scores.mean() * 100, 2)

        return accuracy_percentage, None

    except Exception as e:
        print(f"Error in process_data (Random Forest): {str(e)}")
        raise
