import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as cm

keyword=''
resultsTarget='json_data.json'


doExInhAnalysis=False
doDrugAnalysis=False
doRaw=False
doNormalized=False
doDistillation=True
with open(resultsTarget) as json_file:
    data = json.load(json_file)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


if doDistillation:
    print("Results presented for accuracy overview")
    wholeResultsMatrix=[]
    for method, methodVal in data.items():
        output=[]
        for feature, fval in methodVal.items():
            mid=method + "-" + feature
            print(method + "-" + feature)
            values = np.empty((2, 9))
            for profile, pval in fval.items():
                xid=list(fval.keys()).index(profile)
                for mixing, mixingValue in pval.items():
                    yid = list(pval.keys()).index(mixing)
                    A=mixingValue['cmnorm']
                    A=np.asarray(A)
                    A=np.diag(A[:3,:3])
                    values[yid * 1:(yid + 1) * 1, xid * 3:(xid + 1) * 3]=np.asarray(A)
            output.append(100 * values)
            wholeResultsMatrix.append(100 * values)
        valfinal=np.concatenate(wholeResultsMatrix, axis=0)
        df = pd.DataFrame(data=valfinal)
    print(df)
    df.to_csv(keyword +'_distillation'+ '.csv', index=False, header=False, float_format='%.3f')

if doRaw:
    print("Results presented for raw data export")
    for method, methodVal in data.items():
        output=[]
        for feature, fval in methodVal.items():
            mid=method + "-" + feature
            print(method + "-" + feature)
            values = np.empty((8, 12))
            for profile, pval in fval.items():
                xid=list(fval.keys()).index(profile)
                for mixing, mixingValue in pval.items():
                    yid = list(pval.keys()).index(mixing)
                    A=np.asarray(mixingValue['cmnorm'])
                    B = np.asarray(mixingValue['cm'])
                    values[yid * 4:(yid + 1) * 4, xid * 4:(xid + 1) * 4]=B
            # output.append(100*values)
            output.append(values.astype(int))
        valfinal=np.concatenate(output, axis=0)
        df = pd.DataFrame(data=valfinal)
        print(df)
        df.to_csv(method+keyword+'_rawvalues'+ '.csv',index=False,header=False,float_format='%.3f')
        #exit(2)


if doNormalized:
    print("Results presented for normalized data export")
    for method, methodVal in data.items():
        output=[]
        for feature, fval in methodVal.items():
            mid=method + "-" + feature
            print(method + "-" + feature)
            values = np.empty((8, 12))
            for profile, pval in fval.items():
                xid=list(fval.keys()).index(profile)
                for mixing, mixingValue in pval.items():
                    yid = list(pval.keys()).index(mixing)
                    A=np.asarray(mixingValue['cmnorm'])
                    values[yid * 4:(yid + 1) * 4, xid * 4:(xid + 1) * 4]=A
            output.append(100*values)
        valfinal=np.concatenate(output, axis=0)
        df = pd.DataFrame(data=valfinal)
        print(df)
        df.to_csv(method+keyword+'_relativevalues'+ '.csv',index=False,header=False,float_format='%.3f')




if doDrugAnalysis:
    print("Results presented for drug analysis export")
    fullTable = []
    for method, methodVal in data.items():
        output=[]
        for feature, fval in methodVal.items():
            mid=method + "-" + feature
            print(method + "-" + feature)
            values = np.empty((2, 9))
            for profile, pval in fval.items():
                xid=list(fval.keys()).index(profile)
                for mixing, mixingValue in pval.items():
                    yid = list(pval.keys()).index(mixing)
                    A=np.asarray(mixingValue['cmnorm'])
                    B=np.asarray(mixingValue['cm'])
                    InhaleExhale=B[1:3,1:3]
                    InhaleExhaleAccuracy=np.sum(np.diag(InhaleExhale))/np.sum(InhaleExhale)
                    InhaleExhaleSpecificity=InhaleExhale[1,1]/np.sum(InhaleExhale[:,1])
                    InhaleExhaleSensitivity=InhaleExhale[0, 0] / np.sum(InhaleExhale[:, 0])
                    Drug=np.asarray([B[0,0],np.sum(B[0,1:]),np.sum(B[1:,0]),np.sum(B[1:,1:])]).reshape((2,2))
                    DrugAccuracy=np.sum(np.diag(Drug))/np.sum(Drug)
                    DrugSpecificity=Drug[1,1]/np.sum(Drug[:,1])
                    DrugSensitivity = Drug[0, 0] / np.sum(Drug[:, 0])

                    C=np.asarray([DrugAccuracy,DrugSpecificity,DrugSensitivity])

                    values[yid * 1:(yid + 1) * 1, xid * 3:(xid + 1) * 3]=C
            output.append(100*values)
            fullTable.append(100*values)
    valfinal=np.concatenate(fullTable, axis=0)
    df = pd.DataFrame(data=valfinal)
    print(df)
    df.to_csv(keyword+'_drugAnalysis'+'.csv',index=False,header=False,float_format='%.3f')
    #exit(2)




if doExInhAnalysis:
    print("Results presented for inhalation exhalation analysis export")
    fullTable = []
    for method, methodVal in data.items():
        output=[]
        for feature, fval in methodVal.items():
            mid=method + "-" + feature
            print(method + "-" + feature)
            values = np.empty((2, 9))
            for profile, pval in fval.items():
                xid=list(fval.keys()).index(profile)
                for mixing, mixingValue in pval.items():
                    yid = list(pval.keys()).index(mixing)

                    # A=np.asarray(mixingValue['cmnorm'])
                    B=np.asarray(mixingValue['cm'])
                    InhaleExhale=B[1:3,1:3]
                    InhaleExhaleAccuracy=np.sum(np.diag(InhaleExhale))/np.sum(InhaleExhale)
                    InhaleExhaleSpecificity=InhaleExhale[1,1]/np.sum(InhaleExhale[:,1])
                    InhaleExhaleSensitivity=InhaleExhale[0, 0] / np.sum(InhaleExhale[:, 0])
                    Drug=np.asarray([B[0,0],np.sum(B[0,1:]),np.sum(B[1:,0]),np.sum(B[1:,1:])]).reshape((2,2))
                    DrugAccuracy=np.sum(np.diag(Drug))/np.sum(Drug)
                    DrugSpecificity=Drug[1,1]/np.sum(Drug[:,1])
                    DrugSensitivity = Drug[0, 0] / np.sum(Drug[:, 0])

                    C=np.asarray([InhaleExhaleAccuracy,InhaleExhaleSpecificity,InhaleExhaleSensitivity])

                    values[yid * 1:(yid + 1) * 1, xid * 3:(xid + 1) * 3]=C
            output.append(100*values)
            fullTable.append(100*values)
    valfinal=np.concatenate(fullTable, axis=0)
    df = pd.DataFrame(data=valfinal)
    print(df)
    df.to_csv(keyword+'_InhaleExhaleAnalysis'+'.csv',index=False,header=False,float_format='%.3f')
        #exit(2)