import time
import sys
import re
import numpy as np
import scipy as sp
import csv
from scipy import sparse, io
from scipy.sparse import hstack
from sklearn import pipeline, svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale


from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.externals import joblib

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
      
def applySVM(data,label,modeloutput):
    print;  
    
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
#     sectionMtx = LoadSparseMatrix(ROOTDIR+"train_secion.txt")
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
    '''
    Step 5
    concatenating several columns of vector features into a single data file for training
    labelstack = io.mmread("temp.mtx")
    '''
    data = io.mmread(ROOTDIR+"TRAINDATA.mtx")
    label = np.load(ROOTDIR+"label_train.npy")
    print 'Concatenation Done.'
    
    print str(label.shape)
    print str(data.shape)
    
    
    
    '''
    Step 6
    train it!
    '''
    linear_svm = LinearSVC(C=1.0, class_weight=None, loss='squared_hinge', dual=True, fit_intercept=True,
    intercept_scaling=1, multi_class='ovr', penalty='l2',
    random_state=None, tol=0.0001, verbose=1, max_iter=2000)
    
    data = scale(data, with_mean=False)
    
    linear_svm.fit(data, label)
    joblib.dump(linear_svm, ROOTDIR+'originalTrain_hinge2_1000.pkl') 
#     model = joblib.load(ROOTDIR+'originalTrain_1.pkl')
    
    print 'Trainning Done!'
    scr = linear_svm.score(data, label)
    
    print 'accuracy on the training set is:' + str(scr)
    
if __name__ == '__main__':
    main(sys.argv)
    