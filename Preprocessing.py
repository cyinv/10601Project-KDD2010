import time
import sys

rawTrain = "/home/vincy/course/machine_learning/project/dataset/all_train.txt";
rawTest = "/home/vincy/course/machine_learning/project/dataset/all_test.txt";
train1 = "/home/vincy/course/machine_learning/project/dataset/train1.txt";
train2 = "/home/vincy/course/machine_learning/project/dataset/train2.txt";


## filter raw data, only save 7 columns that we are interested.
## also put the label in the first column.
def filterRawData(filename, savename):
    f = open(filename)
    of = open(savename, "w")
    
    line = f.readline()
    while line:
        parts = line.split("\t")
        
        # ignore the following columns:
        # 0, 4??, 7,8,9,10,11,12,13, 14(label), 15,16,17
        # first output the label:
        of.write(parts[13] + '\t');
        for i in range(len(parts)):
            if i in {0,4,6,7,8,9,10,11,12,13,14,15,16}:
                continue;
            of.write(parts[i] + '\t');
        of.write('\n');
        line = f.readline()
    
    of.close()
    f.close()
    print "done!"
    
    
def buildDictionary(datafile, savefile, column_number):
    f = open(datafile)
    of = open(savefile, "w")
    
    dictionary = {}
    line = f.readline()
    i = 0;
    while line:
        parts = line.split('\t');
        if not dictionary.has_key(parts[column_number]):
            i += 1
            dictionary.update((parts[column_number], i));
            of.write(parts[column_number]+'\n');
        line = f.readline()
    
    f.close()
    of.close()
    print str(i) + " dimension of features, save dictionary to" + savefile + " done!."
    
    
def readDictionary(datafile):
    dic = []; ## use a list to traverse through.
    f = open(datafile)
    line = f.readline()
    while line:
        dic.append(line)
        line = f.readline()
    f.close()
    print "read dictionary from" + datafile + " done!."
    return dic;
    
    
def generateVectorFeature(dic, file):
    
    print; 
    
def main(args):
    filterRawData(rawTrain, train1)
    
if __name__ == '__main__':
    main(sys.argv)
    