import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier

#create average meter
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0
        self.count = 0
    
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#compute binary accuracy
def binary_accuracy(output, target):
    pred = output.cpu() >= 0.5
    accuracy = (pred.int().eq(target.int())).sum()
    accuracy = accuracy * 100 / np.prod(np.array(target.size()))
    return accuracy


#compute pearson correlation coefficient
def compute_pearson_correlation(dataframe):
    df = dataframe.df
    concept_columns = dataframe.attr_cols

    correlations = {
        col: matthews_corrcoef(df["label_index"], df[col])
        for col in concept_columns
    }

    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: abs(item[1])))

    return list(sorted_correlations.keys())

#fit SVM
def fit_svm(X_train, y_train, X_test, y_test):
    svm = LinearSVC()

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions, average = "binary")

    return f1

#compute leakage
def compute_leakage(sorted_concepts, gt_concepts_train, gt_concepts_test, pred_concepts_train, pred_concepts_test, y_train, y_test):
    k = len(sorted_concepts)
    gaps = []

    for i in range(1, len(sorted_concepts) + 1):
        #print(f"Fitting SVM to {i} concept(s)...")  
        concepts_to_use = sorted_concepts[:i]

        gt_C_train = gt_concepts_train[concepts_to_use].values
        gt_C_test = gt_concepts_test[concepts_to_use].values

        predicted_C_train = pred_concepts_train[concepts_to_use].values
        predicted_C_test = pred_concepts_test[concepts_to_use].values

        gt_f1 = fit_svm(gt_C_train, y_train, gt_C_test, y_test)
        #print(f"GT F1-score for {i} concept(s): {gt_f1:.2f}")

        predicted_f1 = fit_svm(predicted_C_train, y_train, predicted_C_test, y_test)
        #print(f"Predicted F1-score for {i} concept(s): {predicted_f1:.2f}")

        gaps.append(predicted_f1-gt_f1)
        #print(f"Gap using {i} concept(s): {predicted_f1-gt_f1:.2f}\n")
    
    
    gaps = np.array(gaps)
    positive_gaps = np.maximum(gaps, 0)
    leakage = np.sum(positive_gaps)/k

    return leakage
    
#compute relevance matrix
def compute_relevance_matrix(predicted_concepts, gt_concepts):
    num_pred = predicted_concepts.shape[1]
    num_gt = gt_concepts.shape[1]

    R = np.zeros((num_pred, num_gt))

    for j in range(num_gt):
        y_j = gt_concepts.iloc[:, j]

        rf = RandomForestClassifier(n_estimators = 10) #fix n = 10 as in the paper
        rf.fit(predicted_concepts.values, y_j)

        importances = rf.feature_importances_
        R[:, j] = importances
    
    return R

#compute disentanglement
def compute_disentanglement(R):
    k = R.shape[1]

    eps = 1e-12
    P = R / (np.sum(R, axis = 1, keepdims=True) + eps) #normalize

    #compute entropy
    entropy = -np.sum(P * np.emath.logn(k, P + eps), axis = 1)
    
    #compute disentanglement
    disentanglement = 1 - entropy

    return disentanglement

#compute DCI
def compute_dci(predicted_concepts, gt_concepts):
    eps = 1e-12

    R = compute_relevance_matrix(predicted_concepts, gt_concepts)
    disentanglement = compute_disentanglement(R)

    weights = np.sum(R, axis = 1)
    total_weight = np.sum(R)

    rho = weights / (total_weight + eps)

    overall_dci = np.sum(rho * disentanglement)

    return overall_dci

#compute OIS
def compute_ois(predicted_concepts, gt_concepts):
    R_gt = compute_relevance_matrix(gt_concepts, gt_concepts)
    R = compute_relevance_matrix(predicted_concepts, gt_concepts)

    k = R.shape[1]

    norm_fn = lambda x: np.linalg.norm(x, ord='fro')

    impurity = norm_fn(np.abs(R_gt - R))
    impurity = impurity / (k / 2)

    return impurity


