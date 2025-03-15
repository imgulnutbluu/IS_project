import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import io, csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# ตั้งค่าชื่อหน้า
st.set_page_config(page_title="IS PROJECT by GULLANUT", layout="wide")

# เมนู Sidebar
menu = st.sidebar.radio("Menu", ["Machine Learning", "Neural Network", "Predicting Video Game Sales", "Predicting Spotify Song Popularity"])

# --- หน้า 1: หน้าแรก ---
if menu == "Machine Learning":
    def show_message():
        st.title("Machine Learning for Best Seller Video Games Prediction")

        st.write("""
        In the Machine Learning section used to predict whether a video game will be a "Best Seller" or not, based on various features of the game such as platform, release year, genre, publisher, and sales in different regions using SVM and Logistic Regression.

        **Prediction Process:**
        1. **Import Data:** The code starts by importing video game data from a CSV file from Kaggle.
        2. **Data Cleaning:**
           - Remove rows with missing data for Year or Publisher.
           - Convert the data type of the Year column to integer.
           - Create a new column named Best_Seller, setting its value to 1 if the global sales (Global_Sales) are greater than 1 million units, and 0 otherwise.
        3. **Select Features and Target:**
           - Define the features used for prediction, which include Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales.
           - Define the target to predict, which is Best_Seller.
        4. **Convert Categorical Data:** Use `pd.get_dummies` to convert categorical data (such as Platform, Genre, Publisher) into numerical values so the Machine Learning model can process it.
        5. **Split the Data:** Split the data into training and testing sets using `train_test_split`.
        6. **Scale the Data:** Use `StandardScaler` to scale the data so that it has a mean of 0 and a standard deviation of 1, which helps improve model performance.
        7. **Create and Train the Model:**
           - Create a SVM model and Logistic Regression model.
           - Train both models using the training set.
        8. **Make Predictions:** Use the trained models to predict whether the video games in the test set will be "Best Sellers."
        9. **Evaluate:**
           - Calculate the accuracy of both models by comparing their predictions with the actual values in the test set.
           - Create a confusion matrix to show the performance of the models in classification.

        **Summary:**
        - **Best Seller Games:** The code defines games with global sales greater than 1 million units as "Best Sellers," which is the target for prediction. These games are considered best sellers based on the defined criteria.
        - **Factors Influencing Sales:** Features like Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, and Other_Sales all affect game sales. The Machine Learning model learns the relationship between these features and sales to predict the likelihood of a game being a "Best Seller."
        - **Further Analysis:** To identify the actual best-selling games, you can analyze the raw data (vgsales.csv) directly by sorting it based on the Global_Sales column from highest to lowest. The games at the top will be the best-sellers.

        **Conclusion:** The result doesn’t directly provide the names of the best-selling games but offers insights into the factors that make a game successful and can be used to predict the likelihood of a game being a "Best Seller." This information can be used for further study to find actual best-selling games.
        """)
    show_message()

# --- หน้า 2: อธิบายการทำงานของโมเดล ---
elif menu == "Neural Network":
    def show_message():
        st.title("Neural Network for Song Popularity Prediction")

        st.write("""
        In the Neural Network (Deep Learning) section, we predict the popularity of songs using the Spotify Songs dataset from Kaggle, with the help of the TensorFlow/Keras library to create the prediction model.

        **Prediction Process:**
        1. **Load and Prepare Data:**
           - Read the Spotify song CSV file (encoding adjustments and NaN value removal may be required).
           - Convert categorical data into numerical values using `LabelEncoder()`.
           - Split the data into Features (X2) and Target (y2).
        2. **Classification or Regression:**
           - Check whether the problem is Classification or Regression by checking the number of unique values in the "popularity" column.
             - If the unique values are ≤ 10 → it's Classification (popularity is categorized).
             - If the unique values are > 10 → it's Regression (popularity is a continuous numeric value).
        3. **Data Preparation:**
           - Split the data into Train (80%) and Test (20%).
           - Use `StandardScaler()` to scale the features so they have a mean of 0 and a standard deviation of 1.
        4. **Create Neural Network Model:**
           - Use `Sequential()` from TensorFlow Keras.
           - The model consists of 3 main layers:
             - **Input Layer:** Accepts the scaled feature count.
             - **Hidden Layers:** 64 and 32 Neurons using ReLU.
             - **Output Layer:**
               - **softmax** (for Classification) → Predict the popularity category of the song.
               - **linear** (for Regression) → Predict the continuous popularity value.
        5. **Train the Model:**
           - Use `Adam()` optimizer.
           - Use `categorical_crossentropy` (for Classification) or `mean_squared_error` (for Regression).
           - Train the model for 20 epochs with a batch size of 64.
        6. **Test and Evaluate:**
           - Use **Accuracy** (for Classification).
           - Use **Mean Absolute Error (MAE)** (for Regression).
        7. **Display Result Graphs:**
           - Loss Graph (Train vs Validation).
           - Accuracy or MAE Graph, depending on the problem type.

        **Summary:**
        This code creates a Deep Learning model that predicts the popularity of songs based on various features from Spotify 🎵📊.

        The prediction results depend on whether it's a Classification or Regression problem:
        - **If it's Classification:** The code shows the **Accuracy** of the model in predicting the popularity category.
          - Example: "Neural Network Accuracy for Spotify Songs: 85.32%" means the model correctly predicted the popularity category 85.32% of the time on the test data.
        - **If it's Regression:** The code shows the **Mean Absolute Error (MAE)**.
          - Example: "Neural Network MAE for Spotify Songs: 2.54" means the predicted popularity values are off from the actual values by an average of 2.54 units.
        """)
    show_message()

# --- หน้า 3: ทำนายยอดขายวิดีโอเกม ---
if menu == "Predicting Video Game Sales":
    st.title("🎮 Predicting Video Game Sales Using SVM and Logistic Regression")
    file_path = "vgsales.csv"
    data = pd.read_csv(file_path)
    
    data_cleaned = data.dropna(subset=['Year', 'Publisher']).copy()
    data_cleaned.loc[:, 'Year'] = data_cleaned['Year'].astype(int)
    data_cleaned.loc[:, 'Best_Seller'] = (data_cleaned['Global_Sales'] > 1).astype(int)
    
    features = ['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    X = pd.get_dummies(data_cleaned[features], drop_first=True)
    y = data_cleaned['Best_Seller']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_option = st.selectbox('Select Models', ['Support Vector Machine', 'Logistic Regression'])
    kernel_option = st.selectbox('Select Kernel for SVM', ['linear', 'rbf', 'poly'], disabled=model_option != 'Support Vector Machine')
    
    def train_and_evaluate(model_option, kernel_option):
        with st.spinner("Training Models ..."):
            if model_option == 'Support Vector Machine':
                model = SVC(kernel=kernel_option, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-Score: {f1:.2f}")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues', ax=ax)
            st.pyplot(fig)
    
    train_and_evaluate(model_option, kernel_option)

# --- หน้า 4: ทำนายความนิยมเพลง Spotify ---
elif menu == "Predicting Spotify Song Popularity":
    st.title("🎵 Predicting Spotify Song Popularity Using Neural Network")
    file_path = "universal_top_spotify_songs.csv"
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
    content = content.replace('\x00', '')
    df2 = pd.read_csv(io.StringIO(content), quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\', on_bad_lines='skip')
    
    df2.dropna(inplace=True)
    for col in df2.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
    
    target_column2 = 'popularity'
    X2 = df2.drop(columns=[target_column2])
    y2 = df2[target_column2]
    
    is_classification = y2.nunique() <= 10
    if is_classification:
        y2 = to_categorical(y2)
    else:
        y2 = y2.values
    
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    scaler2 = StandardScaler()
    X2_train_scaled = scaler2.fit_transform(X2_train)
    X2_test_scaled = scaler2.transform(X2_test)
    
    model = Sequential([
        Input(shape=(X2_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y2_train.shape[1] if is_classification else 1, activation='softmax' if is_classification else 'linear')
    ])
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy' if is_classification else 'mean_squared_error',
                  metrics=['accuracy'] if is_classification else ['mae'])
    
    # ใช้ st.spinner ขณะฝึกโมเดล
    with st.spinner("Training Models ..."):
        history = model.fit(X2_train_scaled, y2_train, epochs=20, batch_size=64, validation_data=(X2_test_scaled, y2_test), verbose=1)
    
    # ใช้ st.spinner ขณะทำนาย
    with st.spinner("Predicting ..."):
        y2_pred = model.predict(X2_test_scaled)
    
    if is_classification:
        y2_pred_classes = y2_pred.argmax(axis=1)
        y2_true = y2_test.argmax(axis=1)
        accuracy_nn = accuracy_score(y2_true, y2_pred_classes)
        st.write(f"Neural Network Accuracy: {accuracy_nn * 100:.2f}%")
    else:
        mae_nn = mean_absolute_error(y2_test, y2_pred)
        st.write(f"Mean Absolute Error: {mae_nn:.2f}")
    
    # แสดงกราฟ Loss over Epochs
    fig = plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(fig)