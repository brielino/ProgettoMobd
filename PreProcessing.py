import pandas as pd
import numpy as num
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

from SelectParamClassifier import svm_param_selection, random_forest_param_selection, decision_tree_param_selection, \
    mlp_param_selection, naive_bayes_param_selection, k_neighbors_classifier_parm_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE

def evaluation(classifier, test_x, test_y):
    # modulo per testare le prestazioni di un classificatore
    print("Risultati Classificatore:\n")
    print("Precision sul test:", metrics.precision_score(test_y, classifier.predict(test_x), average="macro"))
    print()
    print("Accuracy sul test: ", metrics.accuracy_score(test_y, classifier.predict(test_x)))
    print()
    print("F1 sul test:", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    print()
    print("Recall sul test: ", metrics.recall_score(test_y, classifier.predict(test_x), average="macro"))
    print()
    print("Matrice di confusione: \n", metrics.confusion_matrix(test_y, classifier.predict(test_x)))

def calcolo_valori_na(dataset):
    # metodo per individuare quanti sono i valori mancanti nel detaset considerato
    boolean_mask = dataset.isna()
    return boolean_mask.sum(axis=0)

def featureSelection(dataset):
    # metodo per effettuare Feature Selection eliminando le feature non necessarie
    # le variabili False corrispondono agli attributi che elimino
    mask = num.array(
        [False, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, False,
         True, False, True])
    return dataset[:, mask]

def read_label(dataset):
    # metodo per prendere il nome delle label che utilizzerò per la correzione automatica del dataset
    return dataset.columns

def training():
    # metodo per effettuare training dei classifcatori considerati
    # leggo il dataset
    dataset_path = './training_set.csv'
    dataset = pd.read_csv(dataset_path)
    label = read_label(dataset)

    for count in label:
        media = dataset[count].mean()
        dataset[count] = dataset[count].fillna(media)

    # separo gli attributi dal dataset
    x = dataset.iloc[:, 0:20].values
    y = dataset.iloc[:, 20].values
    # separo il dataset in training e test rispettivamente 80% e 20%
    training_x, test_x, training_y, test_y = model.train_test_split(x, y, test_size=0.2, random_state=0)

    # normalizzo i dati
    StandardScaler = preprocessing.MinMaxScaler()
    StandardScaler.fit(training_x)
    training_x = StandardScaler.transform(training_x)
    test_x = StandardScaler.transform(test_x)

    #futureselection
    test_x = featureSelection(test_x)
    training_x = featureSelection(training_x)


    # Sampling
    smote = SMOTE()
    training_x, training_y = smote.fit_resample(training_x, training_y)
    classifier = mlp_param_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for MLP :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    classifier = svm_param_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for SVM :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    classifier = decision_tree_param_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for DecisionTree :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    classifier = random_forest_param_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for RandomForest :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    classifier = naive_bayes_param_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for NaiveBayes :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))
    classifier = k_neighbors_classifier_parm_selection(training_x, training_y, n_folds=10, metric='f1_macro')
    print("F1 for KNeighbors :", metrics.f1_score(test_y, classifier.predict(test_x), average="macro"))

def evaluation_models():
    # metodo per valutare i classificatori dopo aver effettuato training
    # i parametri dei vari classificatori vengono ottenuti dal metodo training() che li stamperà e verranno inseriti manualmente
    dataset_path = './training_set.csv'
    # bisogna inserire il precorso del file da testare
    testset_path = 'testset' # 1)

    #nel caso in cui ci fosse un solo dataset bisognerebbe commentare 1),2),3),4),5) e togliere il commento a 6)

    dataset = pd.read_csv(dataset_path)
    testsetdata = pd.read_csv(testset_path) #2)
    label = read_label(dataset)
    # La separazione 80/20 del dataset non la effettuo in quanto suppongo di avere 2 dataset uno per il training e uno per il testing
    # sostituisco i valori mancanti del testset con la media dei valori nel testingset
    for count in label:
        media = dataset[count].mean()
        testsetdata[count] = testsetdata[count].fillna(media) #3)
        dataset[count] = dataset[count].fillna(media)

    # separo gli attributi dal dataset
    training_x = dataset.iloc[:, 0:20].values
    training_y = dataset.iloc[:, 20].values
    test_x = testsetdata.iloc[:, 0:20].values #4)
    test_y = testsetdata.iloc[:, 20].values #5)
    #training_x, test_x, training_y, test_y = model.train_test_split(training_x, training_y, test_size=0.2, random_state=0) #6)


    # normalizzo i dati
    StandardScaler = preprocessing.MinMaxScaler()
    StandardScaler.fit(training_x)
    training_x = StandardScaler.transform(training_x)
    test_x = StandardScaler.transform(test_x)

    # futureselection
    test_x = featureSelection(test_x)
    training_x = featureSelection(training_x)

    # valutazione modello
    classifier = MLPClassifier(max_iter=10000, activation='relu', hidden_layer_sizes=(100, 50),
                               learning_rate='adaptive', learning_rate_init=0.01, solver='sgd')
    classifier.fit(training_x, training_y)
    print('Risultati MLP')
    evaluation(classifier, test_x, test_y)
    classifier1 = RandomForestClassifier(criterion='entropy', max_depth=100, max_features='log2', min_samples_leaf=1,
                                         min_samples_split=2, n_estimators=400)
    classifier1.fit(training_x, training_y)
    print('Risultati RandomForest')
    evaluation(classifier1, test_x, test_y)

    classifier2 = SVC(C=10, decision_function_shape='ovo', gamma=10, kernel='rbf')
    classifier2.fit(training_x, training_y)
    print('Risultati SVC')
    evaluation(classifier2, test_x, test_y)

    classifier3 = DecisionTreeClassifier(criterion='entropy', max_depth=100, max_features=None, min_samples_leaf=1,
                                         min_samples_split=2, splitter='best')
    classifier3.fit(training_x, training_y)
    print('Risultati DecisionTree')
    evaluation(classifier3, test_x, test_y)

    classifier4 = GaussianNB(priors=None, var_smoothing=1e-9)
    classifier4.fit(training_x, training_y)
    print('Risultati NaiveBayes')
    evaluation(classifier4, test_x, test_y)

    classifier5 = KNeighborsClassifier(algorithm='auto', leaf_size=30, n_neighbors=10, p=3, weights='distance')
    classifier5.fit(training_x, training_y)
    print('Risultati KNeighbors')
    evaluation(classifier, test_x, test_y)

#training()
evaluation_models()
