import time
import sys
import re
import numpy as np
import scipy as sp
import csv
import math
from scipy import sparse, io
from scipy.sparse import hstack
from sklearn import pipeline, svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
from numpy.random import RandomState

import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition
# ROOTDIR = '/home/ubuntu/data/'
ROOTDIR = "/home/vincy/course/machine_learning/project/dataset/"
train1 = ROOTDIR + "train1.txt"
test = ROOTDIR + "test.txt"


def filterRawData(filename, savename):
    '''
    process the raw data, filter out the un-wanted columns.
    '''
    f = open(filename)
    of = open(savename, "w")
    
    line = f.readline().strip('\n')
    while line:
        parts = line.split("\t")
        
        # ignore the following columns:
        # 0, 4??, 7,8,9,10,11,12,13, 14(label), 15,16,17
        # first output the label:
        of.write(parts[13] + '\t');
        for i in range(len(parts)):
            if i in {0,4,6,7,8,9,10,11,12,13,14,15,16}:
                continue;
            of.write(parts[i]);
            if i < 18:
                of.write("\t")
        of.write('\n');
        line = f.readline().strip('\n')
    
    of.close()
    f.close()
    print "done!"
    
    
def buildDictionary(datafile, savefile, column_number):
    '''
    build dictionary for the given column from the data file.
    '''   
    f = open(datafile)
    of = open(savefile, "w")
    
    dictionary = {}
    line = f.readline().strip('\n')
    i = 0;
    while line:
        parts = line.split('\t');
        
        # for ~~ situations. Opportunity cost and the KC
        rparts = parts[column_number].split("~~")
        
        for key in rparts:
            if key == "":
                continue;
            
            if not dictionary.has_key(key):
                i += 1
                dictionary.update({key: i});
                of.write(key+'\n');
                
        line = f.readline().strip('\n')
    
    f.close()
    of.close()
    print str(i-1) + " dimension of features, save dictionary to" + savefile + " done!."
    
    
def readDictionary(datafile):
    '''
    read a dictionary from file 
    '''    
    dic = []; ## use a list to traverse through.
    f = open(datafile)
    f.readline() # ignore the first line.
    line = f.readline().strip('\n')
    while line:
        dic.append(line)
        line = f.readline().strip('\n')
    f.close()
    print "read dictionary of size " + str(len(dic))+ " from" + datafile + " done!."
    return dic;
    
    
def generateVectorFeature(dic, datafile, columnIdx_in_datafile, output, mergeKCOP = None, pattern = None, tokenizer = None):
    '''
    use the dictionary to expand the single column in the data into a sparse vector.
    save the vectorized data to file directly.
    '''
    # the feature in each line of the data file will be extended to a len(dic) vector
    fdata = open(datafile)
    of = open(output, "w")
    
    fdata.readline()    # ignore the first title line.
    
    # first output the name of each dimension of this feature vector
    for item in dic:
        of.write(item + '\t');
    of.write('\n')
    
    sampleCounter = 0; # this is also the row counter
    # output each line
    line = fdata.readline()
    maxHit = 0;
    while line:
        
        if sampleCounter % 20000 == 0:
            print 'vectorizing ' + str(sampleCounter) + 'th sample.'
            
        line = line.strip('\n')
        if line != '':
            curHit = 0;
            cols = line.split("\t")[columnIdx_in_datafile]
            col = cols.split("~~")
            colDic = {}
            idx = 0
            for acol in col:
                if(acol != ""):
                    if tokenizer != None:
                        colDic.update({tokenizer(acol,pattern):idx})
                    else:
                        colDic.update({acol:idx})
                    idx += 1;
            
            # now output the generated vector
            for i in range(len(dic)):
                if(colDic.has_key(dic[i])):
                    if mergeKCOP == None:
                        of.write(str(sampleCounter) + ',' + str(i) + ',1\n')
                    else:
                        # FIXME: this is ugly.....
                        # use this extra condition branch to fix the KC problem.
                        vals = line.split("\t")[mergeKCOP].split('~~')
                        of.write(str(sampleCounter) + ',' + str(i) + ',' +  vals[ colDic.get(dic[i])] + '\n')
                    
                    curHit += 1;
                
            if curHit > maxHit:
                maxHit = curHit
            
        sampleCounter += 1
        line = fdata.readline()
    fdata.close()
    of.close()   
    print str(sampleCounter) + " examples processed. maximum dimension covered: "+ str(maxHit)+"!"; 
    


def generateNumericFeature(datafile, columnIdx_in_datafile, output):
    '''
    @deprecated:  useless...
    simply extract the column elements as numbers.
    used for extract label value.
    '''
    # the feature in each line of the data file will be extended to a len(dic) vector
    fdata = open(datafile)
    of = open(output, "w")
    
    title = fdata.readline()    # ignore the first title line.
    
    # first output the title
    of.write(title);
    
    sampleCounter = 0; # this is also the row counter
    # output each line
    line = fdata.readline()

    while line:
        line = line.strip('\n')
        if line != '':
            item = line.split('\t')[columnIdx_in_datafile]
            of.write(str(sampleCounter) + ',0,' + item +'\n')
                
        sampleCounter += 1
        line = fdata.readline()
    fdata.close()
    of.close()   
    print str(sampleCounter) + " samples processed. With numerical feature in column: " + str(columnIdx_in_datafile); 
    
    
def generateLabel(datafile, columnIdx_in_datafile, output):
    '''
    build a sparse csr matrix, then save them to file.
    '''
    # the feature in each line of the data file will be extended to a len(dic) vector
    fdata = open(datafile)
    fdata.readline()    # ignore the first title line.
    
    row = []
    data = []
    
    sampleCounter = 0; # this is also the row counter

    # output each line
    line = fdata.readline()
    while line:
        line = line.strip('\n')
        if line != '':
            item = line.split('\t')[columnIdx_in_datafile]
            row.append(sampleCounter)
            data.append(int(item))
                
        sampleCounter += 1
        line = fdata.readline()
    fdata.close()
#     lbs = sparse.csr_matrix((data, row))
    np.save(output, data)
    print str(sampleCounter) + " samples processed. With numerical feature in column: " + str(columnIdx_in_datafile); 
        
        
def tokenizeDictionary(inputfile, outputfile):
    '''
    a very naive tokenizer, simply remove all the number characters in the string.
    '''
    f = open(inputfile)
    of = open(outputfile, "w")
    
    line = f.readline();
    of.write(line)
    
    numTokenizer = re.compile(r'\d')
    line = f.readline().strip('\n');
    while line:
        line = line.strip('\n')
        if line != '':
            nline = numTokenizer.sub('', line)
            
            if nline != None:
                nnline = nline.replace(' ','');
                if nnline != None and nnline != '':
                    of.write(nnline + '\n')
        
        line = f.readline();
        
    f.close();
    of.close();
    
    print "tokenize done!."

def tokenizedString(string, pattern):
    '''
    tokenize a string using the given pattern. Replace the matched contents to '', also remove spaces.
    '''
    nline = pattern.sub('', string)
    if nline != None:
        nline = nline.replace(' ','');
        
    return nline

def LoadSparseMatrix(csvfile):
    '''
    Load a sparse matrix from file then return the matrix and the index.
    '''
    val = []
    row = []
    col = []
#     select = []
    f = open(csvfile)
    f.readline() # ignore the first title line.
    reader = csv.reader(f)
    for line in reader:
            row.append( float(line[0]) )
            col.append( float(line[1]) )
            val.append( float(line[2]) )
#             select.append( (float(line[0]), float(line[1])) )
    return sparse.csr_matrix( (val, (row, col)) )#, select


def extractStudentId(dictfile, inputfile, outputfile, testinput, testoutput):    
    '''
    generate student ID sparse matrix
    '''
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 1, outputfile)
    generateVectorFeature(diction, testinput, 1, testoutput)
     
    print 'Student ID Vectorization Done.'

    
def extractProblemHierarchy(dictfile, inputfile, outputfile, testinput, testoutput):    
    '''
    Currently I use the problem + section as a whole.
    '''
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 2, outputfile)
    generateVectorFeature(diction, testinput, 2, testoutput)
    
    print 'ProblemHierarchy Vectorization Done.'

       
def extractProblemName(dictfile, inputfile, outputfile, testinput, testoutput):    
    '''
    Currently I use the problemname directly.
    '''
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 3, outputfile)
    generateVectorFeature(diction, testinput, 3, testoutput)
    
    print 'ProblemName Vectorization Done.'

    
def extractStepName(dictfile, inputfile, outputfile, testinput, testoutput):    
    '''
    Due to the fact that the dimension is too high, i apply a tokenizer to the string first 
    before i use them.
    '''
    tokenizeDictionary(dictfile, ROOTDIR + "dict_step_TKNZED_temp.txt")
    buildDictionary(ROOTDIR + "dict_step_TKNZED_temp.txt", ROOTDIR + "dict_step_TKNZED.txt", 0)
    
    diction = readDictionary(ROOTDIR + "dict_step_TKNZED.txt")
    
    numTokenizer = re.compile(r'\d')
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 4, outputfile, None, numTokenizer, tokenizedString)
    generateVectorFeature(diction, testinput, 4, testoutput, None, numTokenizer, tokenizedString)
    print 'Done.'
    
    
    
def extractKC(dictfile, inputfile, outputfile, testinput, testoutput):    
    '''
    vectorize the KC. 
    '''
    diction = readDictionary(dictfile)
    # don't need to tokenize this vector.
    
    generateVectorFeature(diction, inputfile, 5, outputfile, 6)
    generateVectorFeature(diction, testinput, 5, testoutput, 6)
        
    print 'KC & OP Vectoization Done.'
    
'''
TODO:
This value is now integrated with KC. (as a combination)
'''
def extractOpportunityCount(inputfile, outputfile, testinput, testoutput): 
    print;
  
    
def extractLabels(inputfile, outputfile, testinput, testoutput):
    '''
    output the labels in a single file as a sparse matrix format.
    '''
#     generateNumericFeature(inputfile, 0, outputfile)
#     generateNumericFeature(testinput, 0, testoutput)
    generateLabel(inputfile, 0, outputfile)
    generateLabel(testinput, 0, testoutput)
      
def calcualteRMSE(trueLabel,predLabel):
    total = 0.0
    for i in range(len(trueLabel)):
        total += (trueLabel[i] - predLabel[i]) * (trueLabel[i] - predLabel[i])
    re = math.sqrt(total / len(trueLabel))
    print 'RMSE is: ' + str(re); 
    return re 
    
    
def applySVMWithoutPCA():
    '''
    This is the first experiment. Result is better than before.
    '''
    data = io.mmread(ROOTDIR+"TRAINDATA.mtx")
    label = np.load(ROOTDIR+"label_train.npy")
    testdata = io.mmread(ROOTDIR+"TESTDATA.mtx")
    testLabel = np.load(ROOTDIR + "label_test.npy")
    
    linear_svm = LinearSVC(C=1.0, class_weight=None, loss='hinge', dual=True, fit_intercept=True,
    intercept_scaling=1, multi_class='ovr', penalty='l2',
    random_state=None, tol=0.0001, verbose=1, max_iter=2000)
     
    data = scale(data, with_mean=False)
     
    linear_svm.fit(data, label)
    joblib.dump(linear_svm, ROOTDIR+'originalTrain_hinge_2000.pkl') 
#     linear_svm = joblib.load(ROOTDIR+'originalTrain_hinge_2000.pkl')
    
    print 'Trainning Done!'
    scr = linear_svm.score(data, label)
    print 'accuracy on the training set is:' + str(scr)

    predLabel = linear_svm.predict(data)
    calcualteRMSE(label, predLabel)
    
    scr = linear_svm.score(testdata, testLabel)
    print 'accuracy on the testing set is:' + str(scr)

    predLabel = linear_svm.predict(testdata)
    calcualteRMSE(testLabel, predLabel)
        
      
def applySVMWithPCA():
    '''
    Same as the previous function, just change the file names..
    '''
    data = io.mmread(ROOTDIR+"TRAINDATA.mtx")
    label = np.load(ROOTDIR+"label_train.npy")
    testdata = io.mmread(ROOTDIR+"TESTDATA.mtx")
    testLabel = np.load(ROOTDIR + "label_test.npy")
    
    linear_svm = LinearSVC(C=1.0, class_weight=None, loss='hinge', dual=True, fit_intercept=True,
    intercept_scaling=1, multi_class='ovr', penalty='l2',
    random_state=None, tol=0.0001, verbose=1, max_iter=2000)
     
    data = scale(data, with_mean=False)
     
    linear_svm.fit(data, label)
    joblib.dump(linear_svm, ROOTDIR+'originalTrain_hinge_2000.pkl') 
#     linear_svm = joblib.load(ROOTDIR+'originalTrain_hinge_2000.pkl')
    
    print 'Trainning Done!'
    scr = linear_svm.score(data, label)
    print 'accuracy on the training set is:' + str(scr)

    predLabel = linear_svm.predict(data)
    calcualteRMSE(label, predLabel)
    
    scr = linear_svm.score(testdata, testLabel)
    print 'accuracy on the testing set is:' + str(scr)

    predLabel = linear_svm.predict(testdata)
    calcualteRMSE(testLabel, predLabel)      
        
        
'''
@deprecated: :  Cannot Use PCA Here!!!
'''        
def applyPCA():
    '''
    apply PCA to the data, and analyze the performance.
    After this use SVM to train again.
    '''
    pca = decomposition.PCA()
#     pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    
    kcMtx = LoadSparseMatrix(ROOTDIR+"train_kc.txt")
    
    ###############################################################################
    # Plot the PCA spectrum
    kdense = kcMtx.todense()
    pca.fit(kdense)
    print 'PCA: ' + str(pca.explained_variance_)
    components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ##########
    
    '''
    Lets work on stepName only.
    '''
    sIdMtx = LoadSparseMatrix(ROOTDIR+"train_sId.txt")
    sectionMtx = LoadSparseMatrix(ROOTDIR+"train_section.txt")
    problemMtx = LoadSparseMatrix(ROOTDIR+"train_problem.txt")
    stepMtx = LoadSparseMatrix(ROOTDIR+"train_step.txt")
    
    label = np.load(ROOTDIR+"label_train.npy")
    
    rdata = hstack((sIdMtx, sectionMtx), format='csr')
    rdata = hstack((rdata, problemMtx), format='csr')
    rdata = hstack((rdata, stepMtx), format='csr')
    
    for i in components:
        print 'running PCA for' + str(i)
        pca = decomposition.PCA(n_components=i)
        pca.fit(kdense)
        kpca = pca.transform(kdense)
        
        data = hstack((rdata, kpca), format='csr')
#         io.mmwrite(ROOTDIR+"TRAIN_KMEANS_"+str(i)+".txt", data)
        
        # now train it!
        data = scale(data, with_mean=False)
        lrmodel = linear_model.LogisticRegression(max_iter=1000, penalty='l2', multi_class='ovr', verbose=0)
        
        lrmodel.fit(data, label)
        print 'Trainning Done!'
        scr = lrmodel.score(data, label)
        print 'accuracy on the training set is:' + str(scr)
    
        predLabel = lrmodel.predict(data)
        calcualteRMSE(label, predLabel)
        
        print '************************'
#     plt.figure(1, figsize=(4, 3))
#     plt.clf()
#     plt.axes([.2, .2, .7, .7])
#     plt.plot(pca.explained_variance_, linewidth=2)
#     plt.axis('tight')
#     plt.xlabel('n_components')
#     plt.ylabel('explained_variance_')        
# #     plt.show()
#     plt.savefig('pca.png')

def drawPCA():
    
    pcadata = [  
    6.86332903e+03,   4.85814761e+03,   3.04020699e+03,   2.61233544e+03,
    1.27649886e+03,   8.34322269e+02,   4.90865025e+02,   3.42106526e+02,
    2.91817081e+02,   2.73103893e+02,   2.20392514e+02,   1.90135789e+02,
    9.49357362e+01,   9.39140616e+01,   9.30136328e+01,   9.05535664e+01,
    6.44925021e+01,   5.76760906e+01,   5.02273423e+01,   4.68715308e+01,
    4.26534795e+01,   2.52503897e+01,   2.04923518e+01,   1.62785444e+01,
    1.17038242e+01,   1.13710463e+01,   9.20719062e+00,   9.18713963e+00,
    9.13299210e+00,   8.69136793e+00,   7.95199790e+00,   6.82812076e+00,
    4.67873191e+00,   4.62048427e+00,   3.29182680e+00,   2.69434238e+00,
    2.51511051e+00,   2.40573241e+00,   2.32996711e+00,   2.29887616e+00,
    2.04198774e+00,   2.00281845e+00,   1.88444738e+00,   1.67333923e+00,
    1.57251376e+00,   1.53556362e+00,   1.50486383e+00,   1.50158740e+00,
    1.47319275e+00,   1.44271820e+00,   1.38589061e+00,   1.33004123e+00,
    1.28050021e+00,   1.13767201e+00,   1.11408664e+00,   9.82119325e-01,
    7.99781309e-01,   7.56025102e-01,   6.61580848e-01,   6.30281530e-01,
    6.22760305e-01,   6.01172693e-01,   5.80587629e-01,   5.63185904e-01,
    4.88740312e-01,   4.53192463e-01,   4.33186448e-01,   4.27121313e-01,
    4.16500390e-01,   4.03508039e-01,   4.02601985e-01,   3.33006892e-01,
    3.23821878e-01,   3.05608428e-01,   2.92447329e-01,   2.87547487e-01,
    2.81651321e-01,   2.60593789e-01,   2.02085317e-01,   1.51300039e-01,
    1.10198054e-01,   9.41700262e-02,   6.94692027e-02,   6.42159078e-02,
    5.83705120e-02,   5.70346058e-02,   4.52367317e-02,   3.91098945e-02,
    3.27202601e-02,   3.22305220e-02,   2.34945935e-02,   2.09101285e-02,
    1.77767345e-02,   1.18257383e-02,   1.07423213e-02,   9.75960381e-03,
    6.67300955e-03,   6.40825409e-03,   5.33865789e-03,   5.25320444e-03,
    4.20607601e-03,   3.06733755e-03,   2.92309467e-03,   2.02379224e-03,
    1.09871980e-03,   6.24775055e-04,   4.16174198e-04,   3.20192634e-04,
    2.59245448e-04,   1.07447307e-04,   3.69823368e-06,   1.23503182e-06]
    
    plt.figure(1, figsize=(5, 3))
    plt.clf()
    plt.axes([.1, .1, .8, .8])
    plt.plot(pcadata, linewidth=3)
    plt.axis('tight')
    plt.xlabel('number of components')
    plt.ylabel('variance')        
    plt.title('PCA Variance vs. Components')
    plt.show()
#     plt.savefig('pca.png')
    

'''
@deprecated: :  Cannot Use PCA Here!!!
'''        
def applyKmeans():

    rng = RandomState(0)
    components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#     components = [10]
    '''
    Lets work on stepName only.
    '''
    sIdMtx = LoadSparseMatrix(ROOTDIR+"train_sId.txt")
    sectionMtx = LoadSparseMatrix(ROOTDIR+"train_section.txt")
    problemMtx = LoadSparseMatrix(ROOTDIR+"train_problem.txt")
    stepMtx = LoadSparseMatrix(ROOTDIR+"train_step.txt")
    
    label = np.load(ROOTDIR+"label_train.npy")
    
    rdata = hstack((sIdMtx, sectionMtx), format='csr')
    rdata = hstack((rdata, problemMtx), format='csr')
    rdata = hstack((rdata, stepMtx), format='csr')
    
    kcMtx = LoadSparseMatrix(ROOTDIR + "train_kc.txt")
    
    print 'starting to run kmeans++..'
    
    for i in components:
        km = MiniBatchKMeans(n_clusters=i, tol=1e-3, batch_size=20, max_iter=60, random_state=rng)
        km.fit(kcMtx)
        objscore = km.score(kcMtx)
        
        print 'With ' + str(i) +' components, the object score is ' + str(objscore)
        nkcMtx = km.transform(kcMtx) 
        
#         io.mmwrite(ROOTDIR+"train_step_kmeans_"+str(i)+".txt", nkcMtx)
        
        data = hstack((rdata, nkcMtx), format='csr')
#         io.mmwrite(ROOTDIR+"TRAIN_KMEANS_"+str(i)+".txt", data)
        
        # now train it!
        data = scale(data, with_mean=False)
        lrmodel = linear_model.LogisticRegression(max_iter=1000, penalty='l2', multi_class='ovr', verbose=0)
        
        lrmodel.fit(data, label)
        print 'Trainning Done!'
        scr = lrmodel.score(data, label)
        print 'accuracy on the training set is:' + str(scr)
    
        predLabel = lrmodel.predict(data)
        calcualteRMSE(label, predLabel)
        
        print '************************'

'''
@todo: 
'''        
def applyLogisticRegression():
    '''
    apply logistic regression for the original data set as svm does.
    If this one is better, then we can also use the PCA processed data to train this again.
    '''
    data = io.mmread(ROOTDIR+"TRAINDATA.mtx")
    label = np.load(ROOTDIR+"label_train.npy")
    testdata = io.mmread(ROOTDIR+"TESTDATA.mtx")
    testLabel = np.load(ROOTDIR + "label_test.npy") 
    
    data = scale(data, with_mean=False)
    lrmodel = linear_model.LogisticRegression(max_iter=1000, penalty='l2', multi_class='ovr', verbose=1)
    
    lrmodel.fit(data, label)
    print 'Trainning Done!'
    scr = lrmodel.score(data, label)
    print 'accuracy on the training set is:' + str(scr)

    predLabel = lrmodel.predict(data)
    calcualteRMSE(label, predLabel)
        
def main(args):
    '''
    Step 1
    filter the raw data
    '''
#     filterRawData(ROOTDIR+"all_train.txt", train1)
#     filterRawData(ROOTDIR+"all_test.txt", test)
# 
    '''
    Step 2
    build dictionaries for the categorical features
    '''
#     buildDictionary(train1,ROOTDIR+"dict_sId.txt",1)
#     buildDictionary(train1,ROOTDIR+"dict_section.txt",2)
#     buildDictionary(train1,ROOTDIR+"dict_problem.txt",3)
#     buildDictionary(train1,ROOTDIR+"dict_step.txt",4)
#     buildDictionary(train1, ROOTDIR+"dict_kc.txt",5)
# 
    '''
    Step 3
    process data, extend the columns into feature vectors
    '''
#     extractLabels(train1, ROOTDIR+"label_train.npy", test, ROOTDIR+"label_test.npy")
    
#     extractStudentId(ROOTDIR+"dict_sId.txt", train1, ROOTDIR+"train_sId.txt", test, ROOTDIR+"test_sId.txt")
#     extractProblemHierarchy(ROOTDIR+"dict_section.txt", train1, ROOTDIR+"train_secion.txt",test,ROOTDIR+"test_section.txt")
#     extractProblemName(ROOTDIR+"dict_problem.txt", train1, ROOTDIR+"train_problem.txt",test,ROOTDIR+"test_problem.txt")
#     extractStepName(ROOTDIR+"dict_step.txt",train1,ROOTDIR+"train_step.txt",test,ROOTDIR+"test_step.txt")
#     extractKC(ROOTDIR + "dict_kc.txt", train1, ROOTDIR + "train_kc.txt", test, ROOTDIR + "test_kc.txt")
#     
    
    '''
    Step 4
    further process the data dimensions before concatenate them together.
    '''
#     sIdMtx = LoadSparseMatrix(ROOTDIR+"train_sId.txt")
#     sectionMtx = LoadSparseMatrix(ROOTDIR+"train_section.txt")
#     problemMtx = LoadSparseMatrix(ROOTDIR+"train_problem.txt")
#     stepMtx = LoadSparseMatrix(ROOTDIR+"train_step.txt")
#     kdMtx = LoadSparseMatrix(ROOTDIR + "train_kc.txt")
#     
#    
#     print 'Load Sparse Data Done.'
#     
#     data = hstack((sIdMtx, sectionMtx),format='csr')
#     data = hstack((data, problemMtx),format='csr')
#     data = hstack((data, stepMtx),format='csr')
#     data = hstack((data, kdMtx),format='csr')
    
#     io.mmwrite(ROOTDIR+"TRAINDATA.mtx",data)

#     sIdMtx = LoadSparseMatrix(ROOTDIR+"test_sId.txt")
#     sectionMtx = LoadSparseMatrix(ROOTDIR+"test_section.txt")
#     problemMtx = LoadSparseMatrix(ROOTDIR+"test_problem.txt")
#     stepMtx = LoadSparseMatrix(ROOTDIR+"test_step.txt")
#     kdMtx = LoadSparseMatrix(ROOTDIR + "test_kc.txt")
#       
#      
#     print 'Load Sparse Data Done.'
#       
#     testdata = hstack((sIdMtx, sectionMtx),format='csr')
#     testdata = hstack((testdata, problemMtx),format='csr')
#     testdata = hstack((testdata, stepMtx),format='csr')
#     testdata = hstack((testdata, kdMtx),format='csr')
       
#     io.mmwrite(ROOTDIR+"TESTDATA.mtx",testdata)
    '''
    Step 5
    concatenating several columns of vector features into a single data file for training
    labelstack = io.mmread("temp.mtx")
    '''
#     applySVMWithoutPCA()
    drawPCA()
#     applyLogisticRegression()
#     applyKmeans()

if __name__ == '__main__':
    main(sys.argv)
    