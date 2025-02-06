from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.metrics import silhouette_score


def process_data(X_train, X_test, y_train, y_test, data):
    try:
        # 데이터 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 특성 중요도 계산
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        # 중요 특성만 선택 (상위 2개 특성 선택)
        feature_importances = rf.feature_importances_

        # 중요 특성의 인덱스를 내림차순으로 정렬
        important_feature_indices = feature_importances.argsort()[-2:][::-1]  # 상위 2개 특성 인덱스 선택

        # 상위 2개 중요 특성만 선택하여 새로운 데이터셋 생성
        X_train_important = X_train_scaled[:, important_feature_indices]
        X_test_important = X_test_scaled[:, important_feature_indices]

        # 최적 클러스터 수 결정(엘보우 차트나 자동으로 최적 수를 결정하는 로직 사용 가능)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_train_important)

        # 클러스터 레이블
        labels = kmeans.labels_

        # 클러스터링된 결과 시각화
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_important[:, 0], X_train_important[:, 1], c=labels, cmap='viridis', marker='o',
                    edgecolor='k', s=100)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X',
                    label='Centroids')
        plt.title('KMeans Clustering with Top 2 Important Features')
        plt.xlabel(f'Important Feature 1')
        plt.ylabel(f'Important Feature 2')
        plt.legend()
        plt.grid(True)

        # 클러스터링 결과 이미지 생성
        clustering_img = BytesIO()
        plt.savefig(clustering_img, format='png')
        clustering_img.seek(0)
        clustering_img_base64 = base64.b64encode(clustering_img.getvalue()).decode('utf-8')

        # 정확도는 실루엣 점수로 반환
        accuracy = silhouette_score(X_train_important, labels) * 100

        return round(accuracy, 2), clustering_img_base64

    except Exception as e:
        print(f"Error in process_data (KMeans): {str(e)}")
        raise
