import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
import os
import csv
from sklearn import cross_validation, metrics

#LOAD DATA CLASE
def load_data_class(rel_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)

    with open(abs_file_path, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return your_list


#=========================================================
#=========================================================
if __name__ == "__main__":
    clase1=load_data_class('clases/clase1.csv')
    clase2=load_data_class('clases/clase2.csv')
    clase3=load_data_class('clases/clase3.csv')
    clase4=load_data_class('clases/clase4.csv')
    clase5=load_data_class('clases/clase5.csv')
    clase6=load_data_class('clases/clase6.csv')
    clase7=load_data_class('clases/clase7.csv')
    clase8=load_data_class('clases/clase8.csv')
    clase9=load_data_class('clases/clase9.csv')
    clase10=load_data_class('clases/clase10.csv')
    clase11=load_data_class('clases/clase11.csv')
    clase12=load_data_class('clases/clase12.csv')
    clase13=load_data_class('clases/clase13.csv')
    clase14=load_data_class('clases/clase14.csv')
    clase15=load_data_class('clases/clase15.csv')
    clase16=load_data_class('clases/clase16.csv')
    clase17=load_data_class('clases/clase17.csv')
    clase18=load_data_class('clases/clase18.csv')
    clase19=load_data_class('clases/clase19.csv')
    clase20=load_data_class('clases/clase20.csv')
    clase21=load_data_class('clases/clase21.csv')
    clase22=load_data_class('clases/clase22.csv')
    clase23=load_data_class('clases/clase23.csv')
    clase24=load_data_class('clases/clase24.csv')
    #VECTORxxx[:][NN] -> retonar COLUMNA N de clase1
    #VECTORxxx[NN][:] -> retonar FILA N de clase1

    nn=np.transpose(clase1)
    nn2=np.transpose(clase2)
    nn3=np.transpose(clase3)
    nn4=np.transpose(clase4)
    nn5=np.transpose(clase5)
    nn6=np.transpose(clase6)
    nn7=np.transpose(clase7)
    nn8=np.transpose(clase8)
    nn9=np.transpose(clase9)
    nn10=np.transpose(clase10)
    nn11=np.transpose(clase11)
    nn12=np.transpose(clase12)
    nn13=np.transpose(clase13)
    nn14=np.transpose(clase14)
    nn15=np.transpose(clase15)
    nn16=np.transpose(clase16)
    nn17=np.transpose(clase17)
    nn18=np.transpose(clase18)
    nn19=np.transpose(clase19)
    nn20=np.transpose(clase20)
    nn21=np.transpose(clase21)
    nn22=np.transpose(clase22)
    nn23=np.transpose(clase23)
    nn24=np.transpose(clase24)
    ff=[]
    ff.insert(len(ff),nn[:][0])
    #ff.insert(len(ff),nn[:][1])
    ff.insert(len(ff),nn[:][2])
    #ff.insert(len(ff),nn[:][3])
    #ff.insert(len(ff),nn[:][4])
    ff.insert(len(ff),nn[:][5])
    ff.insert(len(ff),nn[:][6])
    #ff.insert(len(ff),nn[:][7])
    #ff.insert(len(ff),nn[:][8])
    ff.insert(len(ff),nn[:][9])
    #ff.insert(len(ff),nn[:][10])
    #ff.insert(len(ff),nn[:][11])
    ff.insert(len(ff),nn[:][12])
    #ff.insert(len(ff),nn[:][13])
    #ff.insert(len(ff),nn[:][14])
    #ff.insert(len(ff),nn[:][15])
    #ff.insert(len(ff),nn[:][16])
    #ff.insert(len(ff),nn[:][17])
    #ff.insert(len(ff),nn[:][18])
    #ff.insert(len(ff),nn[:][19])
    #ff.insert(len(ff),nn[:][20])
    #ff.insert(len(ff),nn[:][21])
    #ff.insert(len(ff),nn[:][22])
    #ff.insert(len(ff),nn[:][23])
    #ff.insert(len(ff),nn[:][24])

    ff=np.transpose(ff)

    ff2=[]
    ff2.insert(len(ff2),nn2[:][0])
    #ff2.insert(len(ff2),nn2[:][1])
    ff2.insert(len(ff2),nn2[:][2])
    #ff2.insert(len(ff2),nn2[:][3])
    #ff2.insert(len(ff2),nn2[:][4])
    ff2.insert(len(ff2),nn2[:][5])
    ff2.insert(len(ff2),nn2[:][6])
    #ff2.insert(len(ff2),nn2[:][7])
    #ff2.insert(len(ff2),nn2[:][8])
    ff2.insert(len(ff2),nn2[:][9])
    #ff2.insert(len(ff2),nn2[:][10])
    #ff2.insert(len(ff2),nn2[:][11])
    ff2.insert(len(ff2),nn2[:][12])
    #ff2.insert(len(ff2),nn2[:][13])
    #ff2.insert(len(ff2),nn2[:][14])
    #ff2.insert(len(ff2),nn2[:][15])
    #ff2.insert(len(ff2),nn2[:][16])
    #ff2.insert(len(ff2),nn2[:][17])
    #ff2.insert(len(ff2),nn2[:][18])
    #ff2.insert(len(ff2),nn2[:][19])
    #ff2.insert(len(ff2),nn2[:][20])
    #ff2.insert(len(ff2),nn2[:][21])
    #ff2.insert(len(ff2),nn2[:][22])
    #ff2.insert(len(ff2),nn2[:][23])
    #ff2.insert(len(ff2),nn2[:][24])
    ff2=np.transpose(ff2)

    ff3=[]
    ff3.insert(len(ff3),nn3[:][0])
    ff3.insert(len(ff3),nn3[:][2])
    ff3.insert(len(ff3),nn3[:][5])
    ff3.insert(len(ff3),nn3[:][6])
    ff3.insert(len(ff3),nn3[:][9])
    ff3.insert(len(ff3),nn3[:][12])
    ff3=np.transpose(ff3)

    ff4=[]
    ff4.insert(len(ff4),nn4[:][0])
    ff4.insert(len(ff4),nn4[:][2])
    ff4.insert(len(ff4),nn4[:][5])
    ff4.insert(len(ff4),nn4[:][6])
    ff4.insert(len(ff4),nn4[:][9])
    ff4.insert(len(ff4),nn4[:][12])
    ff4=np.transpose(ff4)

    ff5=[]
    ff5.insert(len(ff5),nn5[:][0])
    ff5.insert(len(ff5),nn5[:][2])
    ff5.insert(len(ff5),nn5[:][5])
    ff5.insert(len(ff5),nn5[:][6])
    ff5.insert(len(ff5),nn5[:][9])
    ff5.insert(len(ff5),nn5[:][12])
    ff5=np.transpose(ff5)

    ff6=[]
    ff6.insert(len(ff6),nn6[:][0])
    ff6.insert(len(ff6),nn6[:][2])
    ff6.insert(len(ff6),nn6[:][5])
    ff6.insert(len(ff6),nn6[:][6])
    ff6.insert(len(ff6),nn6[:][9])
    ff6.insert(len(ff6),nn6[:][12])
    ff6=np.transpose(ff6)

    ff7=[]
    ff7.insert(len(ff7),nn7[:][0])
    ff7.insert(len(ff7),nn7[:][2])
    ff7.insert(len(ff7),nn7[:][5])
    ff7.insert(len(ff7),nn7[:][6])
    ff7.insert(len(ff7),nn7[:][9])
    ff7.insert(len(ff7),nn7[:][12])
    ff7=np.transpose(ff7)

    ff8=[]
    ff8.insert(len(ff8),nn8[:][0])
    ff8.insert(len(ff8),nn8[:][2])
    ff8.insert(len(ff8),nn8[:][5])
    ff8.insert(len(ff8),nn8[:][6])
    ff8.insert(len(ff8),nn8[:][9])
    ff8.insert(len(ff8),nn8[:][12])
    ff8=np.transpose(ff8)

    ff9=[]
    ff9.insert(len(ff9),nn9[:][0])
    ff9.insert(len(ff9),nn9[:][2])
    ff9.insert(len(ff9),nn9[:][5])
    ff9.insert(len(ff9),nn9[:][6])
    ff9.insert(len(ff9),nn9[:][9])
    ff9.insert(len(ff9),nn9[:][12])
    ff9=np.transpose(ff9)

    ff10=[]
    ff10.insert(len(ff10),nn10[:][0])
    ff10.insert(len(ff10),nn10[:][2])
    ff10.insert(len(ff10),nn10[:][5])
    ff10.insert(len(ff10),nn10[:][6])
    ff10.insert(len(ff10),nn10[:][9])
    ff10.insert(len(ff10),nn10[:][12])
    ff10=np.transpose(ff10)

    ff11=[]
    ff11.insert(len(ff11),nn11[:][0])
    ff11.insert(len(ff11),nn11[:][2])
    ff11.insert(len(ff11),nn11[:][5])
    ff11.insert(len(ff11),nn11[:][6])
    ff11.insert(len(ff11),nn11[:][9])
    ff11.insert(len(ff11),nn11[:][12])
    ff11=np.transpose(ff11)

    ff12=[]
    ff12.insert(len(ff12),nn12[:][0])
    ff12.insert(len(ff12),nn12[:][2])
    ff12.insert(len(ff12),nn12[:][5])
    ff12.insert(len(ff12),nn12[:][6])
    ff12.insert(len(ff12),nn12[:][9])
    ff12.insert(len(ff12),nn12[:][12])
    ff12=np.transpose(ff12)

    ff13=[]
    ff13.insert(len(ff13),nn13[:][0])
    ff13.insert(len(ff13),nn13[:][2])
    ff13.insert(len(ff13),nn13[:][5])
    ff13.insert(len(ff13),nn13[:][6])
    ff13.insert(len(ff13),nn13[:][9])
    ff13.insert(len(ff13),nn13[:][12])
    ff13=np.transpose(ff13)

    ff14=[]
    ff14.insert(len(ff14),nn14[:][0])
    ff14.insert(len(ff14),nn14[:][2])
    ff14.insert(len(ff14),nn14[:][5])
    ff14.insert(len(ff14),nn14[:][6])
    ff14.insert(len(ff14),nn14[:][9])
    ff14.insert(len(ff14),nn14[:][12])
    ff14=np.transpose(ff14)

    ff15=[]
    ff15.insert(len(ff15),nn15[:][0])
    ff15.insert(len(ff15),nn15[:][2])
    ff15.insert(len(ff15),nn15[:][5])
    ff15.insert(len(ff15),nn15[:][6])
    ff15.insert(len(ff15),nn15[:][9])
    ff15.insert(len(ff15),nn15[:][12])
    ff15=np.transpose(ff15)

    ff16=[]
    ff16.insert(len(ff16),nn16[:][0])
    ff16.insert(len(ff16),nn16[:][2])
    ff16.insert(len(ff16),nn16[:][5])
    ff16.insert(len(ff16),nn16[:][6])
    ff16.insert(len(ff16),nn16[:][9])
    ff16.insert(len(ff16),nn16[:][12])
    ff16=np.transpose(ff16)

    ff17=[]
    ff17.insert(len(ff17),nn17[:][0])
    ff17.insert(len(ff17),nn17[:][2])
    ff17.insert(len(ff17),nn17[:][5])
    ff17.insert(len(ff17),nn17[:][6])
    ff17.insert(len(ff17),nn17[:][9])
    ff17.insert(len(ff17),nn17[:][12])
    ff17=np.transpose(ff17)

    ff18=[]
    ff18.insert(len(ff18),nn18[:][0])
    ff18.insert(len(ff18),nn18[:][2])
    ff18.insert(len(ff18),nn18[:][5])
    ff18.insert(len(ff18),nn18[:][6])
    ff18.insert(len(ff18),nn18[:][9])
    ff18.insert(len(ff18),nn18[:][12])
    ff18=np.transpose(ff18)

    ff19=[]
    ff19.insert(len(ff19),nn19[:][0])
    ff19.insert(len(ff19),nn19[:][2])
    ff19.insert(len(ff19),nn19[:][5])
    ff19.insert(len(ff19),nn19[:][6])
    ff19.insert(len(ff19),nn19[:][9])
    ff19.insert(len(ff19),nn19[:][12])
    ff19=np.transpose(ff19)

    ff20=[]
    ff20.insert(len(ff20),nn20[:][0])
    ff20.insert(len(ff20),nn20[:][2])
    ff20.insert(len(ff20),nn20[:][5])
    ff20.insert(len(ff20),nn20[:][6])
    ff20.insert(len(ff20),nn20[:][9])
    ff20.insert(len(ff20),nn20[:][12])
    ff20=np.transpose(ff20)

    ff21=[]
    ff21.insert(len(ff21),nn21[:][0])
    ff21.insert(len(ff21),nn21[:][2])
    ff21.insert(len(ff21),nn21[:][5])
    ff21.insert(len(ff21),nn21[:][6])
    ff21.insert(len(ff21),nn21[:][9])
    ff21.insert(len(ff21),nn21[:][12])
    ff21=np.transpose(ff21)

    ff22=[]
    ff22.insert(len(ff22),nn22[:][0])
    ff22.insert(len(ff22),nn22[:][2])
    ff22.insert(len(ff22),nn22[:][5])
    ff22.insert(len(ff22),nn22[:][6])
    ff22.insert(len(ff22),nn22[:][9])
    ff22.insert(len(ff22),nn22[:][12])
    ff22=np.transpose(ff22)

    ff23=[]
    ff23.insert(len(ff23),nn23[:][0])
    ff23.insert(len(ff23),nn23[:][2])
    ff23.insert(len(ff23),nn23[:][5])
    ff23.insert(len(ff23),nn23[:][6])
    ff23.insert(len(ff23),nn23[:][9])
    ff23.insert(len(ff23),nn23[:][12])
    ff23=np.transpose(ff23)

    ff24=[]
    ff24.insert(len(ff24),nn24[:][0])
    ff24.insert(len(ff24),nn24[:][2])
    ff24.insert(len(ff24),nn24[:][5])
    ff24.insert(len(ff24),nn24[:][6])
    ff24.insert(len(ff24),nn24[:][9])
    ff24.insert(len(ff24),nn24[:][12])
    ff24=np.transpose(ff24)

    X=np.concatenate((ff, ff2, ff3, ff4, ff5, ff6, ff7, ff8, ff9, ff10, ff11, ff12, ff13, ff14, ff15, ff16, ff17, ff18, ff19, ff20, ff21, ff22, ff23, ff24), axis=0)
    Y=np.zeros(len(X))
    Y[0:len(ff)]=1
    Y[len(ff):len(ff2)]=2
    Y[len(ff2):len(ff3)]=3
    Y[len(ff3):len(ff4)]=4
    Y[len(ff4):len(ff5)]=5
    Y[len(ff6):len(ff7)]=6
    Y[len(ff7):len(ff8)]=7
    Y[len(ff8):len(ff9)]=8
    Y[len(ff9):len(ff10)]=9
    Y[len(ff10):len(ff11)]=10
    Y[len(ff11):len(ff12)]=11
    Y[len(ff12):len(ff13)]=12
    Y[len(ff13):len(ff14)]=13
    Y[len(ff14):len(ff15)]=14
    Y[len(ff15):len(ff16)]=15
    Y[len(ff16):len(ff17)]=16
    Y[len(ff17):len(ff18)]=17
    Y[len(ff18):len(ff19)]=18
    Y[len(ff19):len(ff20)]=19
    Y[len(ff20):len(ff21)]=20
    Y[len(ff21):len(ff22)]=21
    Y[len(ff22):len(ff23)]=22
    Y[len(ff23):len(ff24)]=23
    #Y[len(ff9)]=9

    #X_pred = clf.predict(X)

    #print(X_pred[len(ff):])

    # TRAIN: 60% -- TEST: 40%
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)

    print('-----------')
    print('TRAIN DATA')
    print (X_train.shape)
    print (Y_train.shape)

    print('-----------')
    print('TEST DATA')
    print (X_test.shape)
    print (Y_test.shape)


    clf = svm.SVC(kernel='rbf', gamma=0.1, C=1, cache_size=500).fit(X_train, Y_train)


    #======================
    # SCORE CALCULATION
    #======================
    score = clf.score(X_test, Y_test)

    print("\nAccuracy: %0.2f" % (score))



    #===================
    # CONFUSSION MATRIX
    #===================
    Y_pred = clf.predict(X_test)

    print('\nCONFUSSION MATRIX')
    print(metrics.confusion_matrix(Y_test,Y_pred))


    #=========
    # REPORT
    #=========
    print("\n\nClassification report for classifier %s:\n\n%s\n"
          %
          (clf, metrics.classification_report(Y_test, Y_pred)))

    print("-----------------------------------------------")
    scores = cross_validation.cross_val_score(clf,
                                          X,
                                          Y,
                                          cv=30,
                                          scoring='accuracy') # scoring: accuracy, f1_weighted. etc

    print(scores)
    print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
