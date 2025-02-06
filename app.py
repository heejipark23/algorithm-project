from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import os
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '파일이 없습니다', 400

    file = request.files['file']
    if file.filename == '':
        return '선택된 파일이 없습니다', 400

    if file and file.filename.endswith('.csv'):
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            algorithm = request.form['algorithm']  # 선택된 알고리즘 가져오기
            accuracy, clustering_img = analyze_data(filepath, algorithm)
            return jsonify({'accuracy': accuracy, 'clustering_img': clustering_img})
        except Exception as e:
            return f'분석 중 오류 발생: {str(e)}', 500

    return '허용되지 않는 파일 형식입니다', 400


# 데이터 전처리 함수
def prepare_data(data):
    data = data.copy()
    feature_encoders = {}

    # 특성 데이터 인코딩 (마지막 컬럼 제외)
    for column in data.columns[:-1]:
        if data[column].dtype == 'object':
            feature_encoders[column] = LabelEncoder()
            data[column] = data[column].fillna('missing')
            data[column] = feature_encoders[column].fit_transform(data[column].astype(str))

    # 타깃 데이터 인코딩 (마지막 컬럼)
    target_encoder = None
    if data.iloc[:, -1].dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(data.iloc[:, -1].fillna('missing').astype(str))
    else:
        y = data.iloc[:, -1]

    X = data.iloc[:, :-1]

    return X, y, feature_encoders, target_encoder


def analyze_data(filepath, algorithm):
    try:
        data = pd.read_csv(filepath)

        # 첫 번째 열이 ID인지 확인하고 제거
        first_column = data.columns[0].lower()
        if 'id' in first_column or first_column == 'index':
            data = data.iloc[:, 1:]

        # 수치형 결측치 처리
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        # 데이터 전처리
        X, y, feature_encoders, target_encoder = prepare_data(data)

        # 데이터 정규화
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # 훈련/테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SMOTE를 사용하여 클래스 불균형 해결
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # 알고리즘 처리
        algorithm_module = importlib.import_module(f'algorithms.{algorithm}')
        accuracy, clustering_img = algorithm_module.process_data(X_train, X_test, y_train, y_test, data)

        return accuracy, clustering_img

    except Exception as e:
        print(f"Error in analyze_data: {str(e)}")
        raise


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
