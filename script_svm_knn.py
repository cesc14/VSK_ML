from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

###############################################################################

# =============================================================================
# Kernel choice
# =============================================================================

"""

Here, the kernel for both SVM and SVM-KNN is chosen

"""

kernel = "linear"


# =============================================================================
# Validation setting
# =============================================================================

"""

We define the parameters for the nested CV, including the hyperparameters
of the considered methods

"""

folds = 10   # Folds in CV

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

C_values = [2**i for i in range(-5,5)]
gamma_values = [10**i for i in range(-5, 5)]
degree_values = [1+i for i in range(4)]

p_grid_svm = { 
	"rbf" : {"C": C_values, "kernel": ["rbf"], "gamma" : gamma_values},
    "poly" : {"C": C_values, "kernel": ["poly"], "degree": degree_values},
    "linear" : {"C": C_values, "kernel": ["linear"]}
}

p_grid_knn = {"n_neighbors": [x for x in range(1,12)]}


# =============================================================================
# Simulated dataset generation
# =============================================================================

"""

We generate a two-class dataset (X,y) for testing the different methods

"""

np.random.seed(42)
dataset_size = [500,5]

mean_a  = np.random.uniform(0,20,size=dataset_size[1])
std_a = np.random.uniform(0,2,size=dataset_size[1])
mean_b  = mean_a.copy() + np.random.uniform(0,1,size=dataset_size[1])
std_b = np.random.uniform(0,4.5,size=dataset_size[1])

X_a = np.empty((dataset_size[0]//2,dataset_size[1]))
X_b = X_a.copy()

for i in range(np.size(X_a,axis=0)):
    for j in range(np.size(X_a,axis=1)):
        X_a[i,j] = np.random.normal(mean_a[j],std_a[j])

for i in range(np.size(X_b,axis=0)):
    for j in range(np.size(X_b,axis=1)):
        X_b[i,j] = np.random.normal(mean_b[j],std_b[j])

X = np.vstack((X_a,X_b))

y = np.empty((dataset_size[0],))
y[:len(y)//2]=1
y[len(y)//2:]=-1


# =============================================================================
# Numerical tests 
# =============================================================================

"""

We train and test SVM, KNN and the VSK setting SVM-KNN via nested CV

"""

scaler = StandardScaler()
print("\n")

fold, f1s = 1, []
for train, test in skf.split(X, y):
    
    X_train, X_test = X[train],  X[test]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid_svm[kernel],
                     cv=folds, scoring='f1_weighted') 
    clf.fit(X_train, y[train].ravel())
	
    y_pred = clf.best_estimator_.predict(X_test)   
    f1s.append(f1_score(y[test], y_pred,average = 'weighted'))
        
    fold += 1

print("AVERAGE F1-SCORE SVM:", np.mean(f1s), "+-", np.std(f1s))
print("\n")


fold, f1s = 1, []
for train, test in skf.split(X, y):
    
    X_train, X_test = X[train],  X[test]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = GridSearchCV(KNeighborsClassifier(), param_grid=p_grid_knn,
                     cv=folds, scoring='f1_weighted') 
    clf.fit(X_train, y[train].ravel())
	
    y_pred = clf.best_estimator_.predict(X_test)
    f1s.append(f1_score(y[test], y_pred,average = 'weighted'))

    fold += 1
    
print("AVERAGE F1-SCORE KNN:", np.mean(f1s), "+-", np.std(f1s))
print("\n")


fold, f1s = 1, []
for train, test in skf.split(X, y):
    
    X_train, X_test = X[train],  X[test]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)        

    neigh = GridSearchCV(KNeighborsClassifier(), param_grid=p_grid_knn,
                         cv=folds, scoring='f1_weighted')         
    neigh.fit(X_train,y[train])

    X_train = np.hstack((X_train,neigh.predict_proba(X_train)[:,:-1]))
    X_test = np.hstack((X_test,neigh.predict_proba(X_test)[:,:-1]))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid_svm[kernel
                    ], cv=folds, scoring='f1_weighted') 
    clf.fit(X_train, y[train].ravel())
 	
    y_pred = clf.best_estimator_.predict(X_test)
    f1s.append(f1_score(y[test], y_pred,average = 'weighted'))
    
    fold += 1

print("AVERAGE F1-SCORE SVM-KNN:", np.mean(f1s), "+-", np.std(f1s))
print("\n")
    