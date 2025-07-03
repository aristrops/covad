import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

#compute pearson correlation coefficient
def compute_pearson_correlation(dataframe):
    df = dataframe.df
    concept_columns = dataframe.attr_cols

    correlations = df[concept_columns].apply(lambda col: df["label_index"].corr(col))
    sorted_correlations = correlations.abs().sort_values(ascending = True)

    return sorted_correlations.index.tolist()

#compute leakage
def fit_svm(X_train, y_train, X_test, y_test):
    svm = LinearSVC()

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions, average = "binary")

    return f1

def compute_leakage(sorted_concepts, gt_concepts_train, gt_concepts_test, pred_concepts_train, pred_concepts_test, y_train, y_test):
    k = len(sorted_concepts)
    gaps = []

    for i in range(1, len(sorted_concepts) + 1):
        print(f"Fitting SVM to {i} concept(s)...")  
        concepts_to_use = sorted_concepts[:i]

        gt_C_train = gt_concepts_train[concepts_to_use].values
        gt_C_test = gt_concepts_test[concepts_to_use].values

        predicted_C_train = pred_concepts_train[concepts_to_use].values
        predicted_C_test = pred_concepts_test[concepts_to_use].values

        gt_f1 = fit_svm(gt_C_train, y_train, gt_C_test, y_test)
        print(f"GT F1-score for {i} concept(s): {gt_f1:.2f}")

        predicted_f1 = fit_svm(predicted_C_train, y_train, predicted_C_test, y_test)
        print(f"Predicted F1-score for {i} concept(s): {predicted_f1:.2f}")

        gaps.append(predicted_f1-gt_f1)
        print(f"Gap using {i} concept(s): {predicted_f1-gt_f1:.2f}\n")
    
    
    gaps = np.array(gaps)
    positive_gaps = np.maximum(gaps, 0)
    leakage = np.sum(positive_gaps)/k

    return leakage
    

