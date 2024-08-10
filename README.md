
### Fruit Classification

**Description**: This project demonstrates the application of the K-Nearest Neighbors (KNN) algorithm to classify various types of fruits based on their features such as mass, width, and height. The goal is to build a model that can accurately predict the type of fruit from these measurements.

#### Project Overview

Fruit classification is a common problem in machine learning that involves identifying the type of fruit based on its physical attributes. In this project, we use the K-Nearest Neighbors (KNN) algorithm, a simple yet powerful classification technique, to achieve this.

Key Components

1. **Dataset**:
   - **Source**: The dataset is provided in a CSV file named `fruit_dataset.csv`.
   - **Features**: 
     - `mass`: Mass of the fruit (in grams).
     - `width`: Width of the fruit (in centimeters).
     - `height`: Height of the fruit (in centimeters).
   - **Label**: `fruit_label` indicating the type of fruit (e.g., apple, orange, banana).

2. **Data Preprocessing**:
   - **Loading Data**: The dataset is loaded using Pandas.
   - **Exploratory Data Analysis (EDA)**: Visualizations are created to understand the distribution of features and their relationships with the labels.

3. **Model Implementation**:
   - **Algorithm**: K-Nearest Neighbors (KNN) is used for classification.
   - **Parameters**: The number of neighbors, `k`, is set to 5.
   - **Distance Metric**: Euclidean distance is used to measure the similarity between data points.

4. **Training and Testing**:
   - **Data Splitting**: The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.
   - **Model Training**: The KNN model is trained using the training dataset.
   - **Model Evaluation**: The accuracy of the model is evaluated on both the training and testing datasets.

5. **Visualization**:
   - **Scatter Plot**: A scatter plot is created to visualize the distribution of fruits based on their width and height. Different colors represent different fruit types.

6. **Results**:
   - **Accuracy**: The performance of the model is assessed, showing how well it can classify fruits based on the given features.
   - **Predictions**: The model can predict the type of fruit for new data points.

