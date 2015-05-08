import time
import sys
import re
import numpy as np
import csv
from scipy import sparse

ROOTDIR = "/home/vincy/course/machine_learning/project/dataset/"
train1 = ROOTDIR + "train1.txt"
test = ROOTDIR + "test.txt"

'''
process the raw data, filter out the un-wanted columns.
'''
def filterRawData(filename, savename):
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
    
'''
build dictionary for the given column from the data file.
'''   
def buildDictionary(datafile, savefile, column_number):
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
    
    
'''
read a dictionary from file 
'''    
def readDictionary(datafile):
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
    
    
'''
use the dictionary to expand the single column in the data into a sparse vector.
save the vectorized data to file directly.
'''
def generateVectorFeature(dic, datafile, columnIdx_in_datafile, output, mergeKCOP = None, pattern = None, tokenizer = None):
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
            
            colCounter = 0;
            
            # now output the generated vector
            for i in range(len(dic)):
                if(colDic.has_key(dic[i])):
                    if mergeKCOP == None:
                        of.write(str(sampleCounter) + ',' + str(colCounter) + ',1\n')
                    else:
                        # FIXME: this is ugly.....
                        # use this extra condition branch to fix the KC problem.
                        vals = line.split("\t")[mergeKCOP]
                        of.write(str(sampleCounter) + ',' + str(colCounter) + ',' +  vals[ colDic.get(dic[i])] + '\n')
                    
                    curHit += 1;
                colCounter += 1;
                
            if curHit > maxHit:
                maxHit = curHit
            
        sampleCounter += 1
        line = fdata.readline()
    fdata.close()
    of.close()   
    print str(sampleCounter) + " examples processed. maximum dimension covered: "+ str(maxHit)+"!"; 
    

'''
simply extract the column elements as numbers.
used for extract label value.
'''
def generateNumericFeature(datafile, columnIdx_in_datafile, output):
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
    
        
'''
a very naive tokenizer, simply remove all the number characters in the string.
'''
def tokenizeDictionary(inputfile, outputfile):
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

'''
tokenize a string using the given pattern. Replace the matched contents to '', also remove spaces.
'''
def tokenizedString(string, pattern):
    nline = pattern.sub('', string)
    if nline != None:
        nline = nline.replace(' ','');
        
    return nline

'''
Load a sparse matrix from file then return the matrix and the index.
'''
def LoadSparseMatrix(csvfile):
        val = []
        row = []
        col = []
        select = []
        f = open(csvfile)
        f.readline() # ignore the first title line.
        reader = csv.reader(f)
        for line in reader:
                row.append( float(line[0]) )
                col.append( float(line[1]) )
                val.append( float(line[2]) )
                select.append( (float(line[0]), float(line[1])) )
        return sparse.csr_matrix( (val, (row, col)) ), select

'''
generate student ID sparse matrix
'''
def extractStudentId(dictfile, inputfile, outputfile, testinput, testoutput):    
    
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 1, outputfile)
    generateVectorFeature(diction, testinput, 1, testoutput)
     
    print 'Student ID Vectorization Done.'

'''
Currently I use the problem + section as a whole.
'''
def extractProblemHierarchy(dictfile, inputfile, outputfile, testinput, testoutput):    
    
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 2, outputfile)
    generateVectorFeature(diction, testinput, 2, testoutput)
    
    print 'ProblemHierarchy Vectorization Done.'

'''
Currently I use the problemname directly.
'''
def extractProblemName(dictfile, inputfile, outputfile, testinput, testoutput):    
       
    diction = readDictionary(dictfile)
    
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 3, outputfile)
    generateVectorFeature(diction, testinput, 3, testoutput)
    
    print 'ProblemName Vectorization Done.'

'''
Due to the fact that the dimension is too high, i apply a tokenizer to the string first 
before i use them.
'''
def extractStepName(dictfile, inputfile, outputfile, testinput, testoutput):    
    
    tokenizeDictionary(dictfile, ROOTDIR + "tokenizedstpDict.txt")
    diction = readDictionary(ROOTDIR + "tokenizedstpDict.txt")
    
    numTokenizer = re.compile(r'\d')
    # don't need to tokenize this vector.
    generateVectorFeature(diction, inputfile, 4, outputfile, numTokenizer, tokenizedString)
    generateVectorFeature(diction, testinput, 4, testoutput, numTokenizer, tokenizedString)
    print 'Done.'
    
    
'''
vectorize the KC. 
'''
def extractKC(dictfile, inputfile, outputfile, testinput, testoutput):    
    
    diction = readDictionary(dictfile)
    # don't need to tokenize this vector.
    
    generateVectorFeature(diction, inputfile, 5, 6,outputfile)
    generateVectorFeature(diction, testinput, 5, 6,testoutput)
        
    print 'KC & OP Vectoization Done.'
    
'''
TODO:
This value is now integrated with KC. (as a combination)
'''
def extractOpportunityCount(inputfile, outputfile, testinput, testoutput): 
    print;
  
'''
output the labels in a single file as a sparse matrix format.
'''
def extractLabels(inputfile, outputfile, testinput, testoutput):
    
    generateNumericFeature(inputfile, 0, outputfile)
    generateNumericFeature(testinput, 0, testoutput)
    
      
def main(args):
    '''
    Step 1
    filter the raw data
    '''
    filterRawData(ROOTDIR+"all_train.txt", test)
    filterRawData(ROOTDIR+"all_test.txt", train1)

    '''
    Step 2
    build dictionaries for the categorical features
    '''
    buildDictionary(train1,ROOTDIR+"sIdDict.txt",1)
    buildDictionary(train1,ROOTDIR+"sectionDict.txt",2)
    buildDictionary(train1,ROOTDIR+"problemDict.txt",3)
    buildDictionary(train1,ROOTDIR+"stpDict.txt",4)
    buildDictionary(train1, ROOTDIR+"kcDict.txt",5)

    '''
    Step 3
    process data, extend the columns into feature vectors
    '''
    extractLabels(train1, ROOTDIR+"trainLabel.txt", test, ROOTDIR+"testlabel.txt")
    
    extractStudentId(ROOTDIR+"sIdDict.txt", train1, ROOTDIR+"sIDtrain.txt", test, ROOTDIR+"sIDtest.txt")
    extractProblemHierarchy(ROOTDIR+"sectionDict.txt", train1, ROOTDIR+"seciontrain.txt",test,ROOTDIR+"sectiontest.txt")
    extractProblemName(ROOTDIR+"problemDict.txt", train1, ROOTDIR+"problemtrain.txt",test,ROOTDIR+"problemtest.txt")
    extractStepName(ROOTDIR+"stpDict.txt",train1,ROOTDIR+"stptrain.txt",test,ROOTDIR+"stptest.txt")
    extractKC(ROOTDIR + "kcDict.txt", train1, ROOTDIR + "kcTrain.txt", test, ROOTDIR + "kcTest.txt")
    
    
    '''
    Step 4
    concatenating several columns of vector features into a single data file
    '''
    
    '''
    Step 5
    train it!
    '''
#     generateVectorFeature(dic, test, 5, ROOTDIR+"sparseKCtest.txt")
#     sparsemtx, select = LoadSparseMatrix(ROOTDIR+"sparseKCtest.txt")
    
#     tokenizeDictionary(stpdict,temp)
#     buildDictionary(temp,stpdictTKNZD,0)
#     extractOpportunityCount(train1, test, ROOTDIR+"kcDict.txt", ROOTDIR+"kcVecTrain.txt", ROOTDIR+"kcVecTest.txt")
#     LoadSparseMatrix(temp)
                   
if __name__ == '__main__':
    main(sys.argv)
    