import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from numpy import expand_dims
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


# face_emb=[]
def get_data(datasetname):
    name=[]
    directory = os.path.join("C:/Users/pnc/Documents/impact.ai/mask-feature-exp/dataset/"+datasetname+"/")
    
    frames=[]
    for root,dirs,files in os.walk(directory):
        # print(dirs)
        
        for direc in dirs:
            name.append(direc)
            fil=os.path.join(directory + direc)
            
            
            a=[]
            for roots,dirss,filess in os.walk(fil): 
                for file in filess:
                    a.append(file)
                frames.append(a)
    

    img_arr=[]
    img_name=[]
    img_label=[]
    an=0
    for i in range(len(frames)):
        len_el=len(frames[i])
        temp=frames[i]
        for j in range(len_el):
        
            img_name.append(temp[j])
            img_label.append(name[i])
    
            st=os.path.join(directory + name[i])
            a=st+"/"+temp[j]
            
            tr=np.load(a)
            
            img_arr.append(tr)
    img_arr=np.array(img_arr)
    img_label=np.array(img_label)

    return img_arr,img_label


# def svc_classifier(trainX,trainY,testX,testY,img_arr,img_label):
#     model = svm.SVC(kernel='linear')
#     nsamples, nx, ny = trainX.shape
#     trainX = trainX.reshape((nsamples,nx*ny))
#     nsamples1, nx1, ny1 = testX.shape
#     testX = testX.reshape((nsamples1,nx1*ny1))
#     nsamples2, nx2, ny2 = img_arr.shape
#     img_arr = img_arr.reshape((nsamples2,nx2*ny2))
    
#     model.fit(trainX, trainY)
#     yhat_train=model.predict(trainX)
#     print(yhat_train)
#     yhat_test = model.predict(testX)
#     print(yhat_test)
    
#     score_train = accuracy_score(trainY, yhat_train)
#     score_test = accuracy_score(testY, yhat_test)
#     print(score_train)
#     print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
    
#     yhat_class = model.predict(img_arr)
    
#     print(yhat_class)
    
#     print(trainY)
#     print(yhat_train)
#     conf_matrix(img_label,yhat_class)
def svc_classifier_fit(X,Y):
    model = svm.SVC(kernel='linear')
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def k_neighbor_fit(X,Y):
    model = KNeighborsClassifier(3)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def gausian_process_fit(X,Y):
    model =  GaussianProcessClassifier(1.0 * RBF(1.0))
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def Decision_tree_fit(X,Y):
    model = DecisionTreeClassifier(max_depth=5)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def MLPClassifier_fit(X,Y):
    model = MLPClassifier(alpha=1, max_iter=1000)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def RandomForest_fit(X,Y):
    model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def Adaboost_fit(X,Y):
    model = AdaBoostClassifier()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def GaussianNB_fit(X,Y):
    model = GaussianNB()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model
def QDA_fit(X,Y):
    model = QuadraticDiscriminantAnalysis()
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    model.fit(X, Y)
    return model

def predict(model,X,Y):
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    predicted=model.predict(X)
    score_ = accuracy_score(Y,predicted)
    conf_matrix(Y,predicted)

    
def conf_matrix(trainY,yhat_train):
    b=confusion_matrix(trainY,yhat_train)
    print(trainY)
    print(yhat_train)
    p_s=precision_score(trainY,yhat_train,average='micro')
    # r_s=recall_score(trainY,yhat_train)
    print("confusion matrix")
    print(b)
    print("precision score")
    print(p_s)
    # print(r_s)

img_array,img_label = get_data("megha_mask")
trainX,trainY=get_data("train")
testX,testY=get_data("test")
svc_model=svc_classifier_fit(trainX,trainY)
print("no-mask-svc")
predict(svc_model,img_array,img_label)
print("mask-svc")
predict(svc_model,testX,testY)
kNeighbor_model=k_neighbor_fit(trainX,trainY)
print("no-mask-Kneighbor")
predict(kNeighbor_model,img_array,img_label)
print("mask-Kneighbor")
predict(kNeighbor_model,testX,testY)
gaussian_process_model=gausian_process_fit(trainX,trainY)
print("no-mask-gaussian-process")
predict(gaussian_process_model,img_array,img_label)
print("mask-gaussian-process")
predict(gaussian_process_model,testX,testY)
decision_tree_model=Decision_tree_fit(trainX,trainY)
print("no-mask-decision-tree")
predict(decision_tree_model,img_array,img_label)
print("mask-decision-tree")
predict(decision_tree_model,testX,testY)
MLP_model=MLPClassifier_fit(trainX,trainY)
print("no-mask-MLP")
predict(MLP_model,img_array,img_label)
print("mask-MLP")
predict(MLP_model,testX,testY)
random_model=RandomForest_fit(trainX,trainY)
print("no-mask-random")
predict(random_model,img_array,img_label)
print("mask-random")
predict(random_model,testX,testY)
adaboost_model=Adaboost_fit(trainX,trainY)
print("no-mask-adaboost")
predict(adaboost_model,img_array,img_label)
print("mask-adaboost")
predict(adaboost_model,testX,testY)
# GaussianNB_model=GaussianNB(trainX,trainY)
# print("no-mask-GaussianNB")
# predict(GaussianNB_model,img_array,img_label)
# print("mask-GaussianNB")
# predict(GaussianNB_model,testX,testY)
QDA_model=QDA_fit(trainX,trainY)
print("no-mask-QDA")
predict(QDA_model,img_array,img_label)
print("mask-QDA")
predict(QDA_model,testX,testY)

