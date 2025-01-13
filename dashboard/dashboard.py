import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# Page Title
st.title("Enhanced Model Performance Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to", ["Overview", "Performance Metrics", "Feature Importance", "Error Analysis", "Threshold Tuning"]
)

# Sample Data (Replace with actual data)
acc = [0.7, 0.8, 0.87]
loss = [0.6, 0.5, 0.4]
cm = [[50, 2], [3, 45]]  # Confusion Matrix
y_true = [0, 0, 1, 1, 0, 1, 0, 1]  # True labels
y_pred = [0, 0, 1, 1, 1, 0, 0, 1]  # Predicted labels
y_prob = [0.1, 0.2, 0.8, 0.9, 0.7, 0.4, 0.3, 0.9]  # Predicted probabilities
feature_importances = {"Feature A": 0.4, "Feature B": 0.3, "Feature C": 0.2, "Feature D": 0.1}

# Metrics to Display
metrics = ["precision", "recall", "f1-score"]

# Section: Overview
if section == "Overview":
    st.header("Model Training Overview")
    st.subheader("Accuracy and Loss Curves")
    fig, ax = plt.subplots()
    ax.plot(acc, label="Accuracy")
    ax.plot(loss, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Section: Performance Metrics
elif section == "Performance Metrics":
    st.header("Classification Metrics")
    report = classification_report(y_true, y_pred, output_dict=True)
    class_labels = [label for label in report.keys() if label not in ["accuracy", "macro avg", "weighted avg"]]

    st.subheader("Classification Report")
    st.table({metric: [report[label][metric] for label in class_labels] for metric in metrics})

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# Section: Feature Importance
elif section == "Feature Importance":
    st.header("Feature Importance")
    fig, ax = plt.subplots()
    features = list(feature_importances.keys())
    importances = list(feature_importances.values())
    ax.barh(features, importances, color="skyblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Section: Error Analysis
elif section == "Error Analysis":
    st.header("Error Analysis")
    st.subheader("Misclassified Samples")
    misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    st.write("Indices of misclassified samples:", misclassified_indices)

    st.subheader("Residual Plot (Regression Example)")
    y_actual = [3, -0.5, 2, 7]
    y_predicted = [2.5, 0.0, 2, 8]
    residuals = np.array(y_actual) - np.array(y_predicted)
    fig, ax = plt.subplots()
    ax.scatter(y_actual, residuals, alpha=0.7)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

# Section: Threshold Tuning
elif section == "Threshold Tuning":
    st.header("Classification Threshold Tuning")
    threshold = st.slider("Select threshold", 0.0, 1.0, 0.5, step=0.01)
    y_adjusted_pred = [1 if prob >= threshold else 0 for prob in y_prob]

    st.subheader(f"Adjusted Predictions at Threshold: {threshold}")
    st.write("Adjusted Predictions:", y_adjusted_pred)

    # Update Confusion Matrix
    adjusted_cm = confusion_matrix(y_true, y_adjusted_pred)
    fig, ax = plt.subplots()
    sns.heatmap(adjusted_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Metrics at Selected Threshold")
    report_adjusted = classification_report(y_true, y_adjusted_pred, output_dict=True)
    class_labels = [label for label in report_adjusted.keys() if label not in ["accuracy", "macro avg", "weighted avg"]]
    st.table({metric: [report_adjusted[label][metric] for label in class_labels] for metric in metrics})