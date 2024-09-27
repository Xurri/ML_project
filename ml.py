# Импорт необходимых библиотек
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import sklearn
sklearn.set_config(transform_output='pandas')

# Настройка страницы Streamlit
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Заголовок приложения
st.title("🏡 House Price Prediction App")
st.write("Upload your test dataset and get house price predictions using a pre-trained CatBoost model.")

# Загрузка модели
model = joblib.load('cat_boost_pipline.pkl')

# Функция для предобработки входных данных
def preprocess_input_data(input_data, numeric_columns, categorical_columns):
    # Заполнение пропусков
    input_data[numeric_columns] = input_data[numeric_columns].fillna(input_data[numeric_columns].median())
    input_data[categorical_columns] = input_data[categorical_columns].fillna("Missing")

    # Применение скейлера к числовым данным
    scaler = StandardScaler()
    input_data[numeric_columns] = scaler.fit_transform(input_data[numeric_columns])

    # Применение TargetEncoder к категориальным данным
    encoder = TargetEncoder()
    input_data[categorical_columns] = encoder.fit_transform(input_data[categorical_columns], np.zeros(len(input_data)))

    return input_data

# Функция для выполнения предсказаний
def predict_and_evaluate(model, X):
    # Выполнение предсказания
    predictions = model.predict(X)
    return predictions

# Определение загружаемого файла пользователем
uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

# Если загружен файл, выполняем предсказание
if uploaded_file is not None:
    # Чтение загруженного файла
    input_data = pd.read_csv(uploaded_file)

    # Удаление столбца 'Id', если он есть
    if 'Id' in input_data.columns:
        input_data = input_data.drop(['Id'], axis=1)

    # Определение числовых и категориальных колонок
    numeric_columns = input_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = input_data.select_dtypes(include=['object']).columns.tolist()

    # Обработка входных данных
    input_data = preprocess_input_data(input_data, numeric_columns, categorical_columns)

    # Отображаем загруженные данные
    st.subheader("Uploaded Data Preview")
    st.write(input_data.head())

    # Выполнение предсказаний
    predictions = predict_and_evaluate(model, input_data)

    # Преобразование логарифмированных данных обратно к исходным значениям
    predictions = np.expm1(predictions)

    # Создание DataFrame с результатами
    results_df = pd.DataFrame()
    results_df['Predicted SalePrice'] = predictions

    # Отображение результатов
    st.subheader("Prediction Results")
    st.write(results_df.head())

    # Добавление возможности загрузить результаты предсказаний
    st.download_button(label="Download Predictions as CSV",
                       data=results_df.to_csv(index=False).encode('utf-8'),
                       file_name='house_price_predictions.csv',
                       mime='text/csv')

    # Визуализация результатов предсказаний
    st.subheader("Visualization of Predictions")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # График 1: Гистограмма предсказанных значений
    sns.histplot(predictions, kde=True, color='skyblue', ax=ax[0])
    ax[0].set_title("Distribution of Predicted House Prices")
    ax[0].set_xlabel("Predicted Sale Price")

    # # График 2: Scatter plot реальных против предсказанных (если есть реальные значения)
    # if 'SalePrice' in input_data.columns:
    #     true_prices = np.expm1(input_data['SalePrice'])  # Обратное преобразование, если SalePrice в данных
    #     sns.scatterplot(x=true_prices, y=predictions, ax=ax[1], color='purple')
    #     ax[1].set_xlabel("Actual Sale Price")
    #     ax[1].set_ylabel("Predicted Sale Price")
    #     ax[1].set_title("Actual vs Predicted Sale Prices")

    # Отображение графиков
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file for prediction.")
