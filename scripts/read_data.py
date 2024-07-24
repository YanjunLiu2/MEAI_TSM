import xlrd
import os
import numpy as np
file_path = os.path.join(os.path.dirname(__file__), '..', 'resources')

def generate_data(group, file_loc=file_path):
    # X data generation
    excel = xlrd.open_workbook(file_loc+"/16Mdata1.xls")

    all_sheet = excel.sheets()

    # read all the data
    database = all_sheet[0]
    allresult = database.col_values(0)[3:]
    allmatname = database.col_values(1)[3:]
    alldsq = database.col_values(2)[3:]
    alldv = database.col_values(3)[3:]
    allsqatom = database.col_values(8)[3:]
    for i in range(len(allsqatom)):
        if allsqatom[i] == 'D':
            allsqatom[i] = 'H'
    alladata = database.col_values(10)[3:]
    allcdata = database.col_values(11)[3:]
    allscdata = database.col_values(12)[3:]
    structure = database.col_values(13)[3:]

    # generate subgroup data based on different structure types
    result = [[] for n in range(len(group))]
    matname = [[] for n in range(len(group))]
    dsq = [[] for n in range(len(group))]
    dv = [[] for n in range(len(group))]
    sqatom = [[] for n in range(len(group))]
    adata = [[] for n in range(len(group))]
    cdata = [[] for n in range(len(group))]


    ordnumber=[]
    for n in range(len(structure)):
        gui = 1
        if (allresult[n] == 'yes'):
            gui = 0
            for ll in range(2):
                for lll in range(len(group[ll])):
                    if structure[n] == group[ll][lll]:
                        gui = 1
            if gui == 0:
                ordnumber.append(n)
                #county = county + 1
        for k in range(len(group)):
            #tip=0
            for l in range(len(group[k])):
                if structure[n] == group[k][l]:
                    result[k].append(allresult[n])
                    matname[k].append(allmatname[n])
                    dsq[k].append(alldsq[n])
                    dv[k].append(alldv[n])
                    sqatom[k].append(allsqatom[n])
                    adata[k].append(alladata[n])
                    cdata[k].append(allcdata[n])
                    tip=1

            """if (k == 1) and (gui == 0):
                result[k].append(allresult[n])
                matname[k].append(allmatname[n])
                dsq[k].append(alldsq[n])
                dv[k].append(alldv[n])
                sqatom[k].append(allsqatom[n])
                adata[k].append(alladata[n])
                cdata[k].append(allcdata[n])"""
    print(len(ordnumber),'ordnumber')
    print(ordnumber)
    """ccnn=0
    for n in range(len(ordnumber)):
        if (n!=8)and(n!=10)and(n!=11)and(n!=16):
            ccnn+=1
            result[1].append(allresult[ordnumber[n]])
            matname[1].append(allmatname[ordnumber[n]])
            dsq[1].append(alldsq[ordnumber[n]])
            dv[1].append(alldv[ordnumber[n]])
            sqatom[1].append(allsqatom[ordnumber[n]])
            adata[1].append(alladata[ordnumber[n]])
            cdata[1].append(allcdata[ordnumber[n]])
    print(ccnn,'ccnn')"""

    # read atomic database
    excel1 = xlrd.open_workbook(file_loc+"/atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(file_loc+"/xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(file_loc+"/Econfig.xls")
    all_sheet3 = excel3.sheets()
    edata = all_sheet3[0]
    ele2 = edata.col_values(0)[1:]
    valence = edata.col_values(1)[1:]

    Xtot = []
    Ytot = []
    composition = []
    elist = []
    for i in range(len(group)):
        Xtot.append([])
        Ytot.append([])
        composition.append([])
        elist.append([])

    for n in range(len(group)):
        labels = [x[0] == "y" for x in result[n]]
        for i in range(len(matname[n])):
            elements = []
            ratio = []
            for j in range(len(matname[n][i])):
                if (j < len(matname[n][i]) - 1) and (matname[n][i][j] == r')') and (
                        ord(matname[n][i][j + 1]) > 47) and (
                        ord(matname[n][i][j + 1]) < 58):
                    print(i, matname[n][i])
                if (ord(matname[n][i][j]) > 64) & (ord(matname[n][i][j]) < 91):
                    ele = matname[n][i][j]
                    if j == (len(matname[n][i]) - 1):
                        rate = 1.
                    if j < (len(matname[n][i]) - 1):
                        if (ord(matname[n][i][j + 1]) > 96) & (ord(matname[n][i][j + 1]) < 123):
                            ele = ele + matname[n][i][j + 1]
                            if (j + 2 == len(matname[n][i])):
                                rate = 1.
                            elif ((ord(matname[n][i][j + 2]) > 64) & (ord(matname[n][i][j + 2]) < 91)) or (
                                    matname[n][i][j + 2] == r' ') or (matname[n][i][j + 2] == r'(') or (
                                    matname[n][i][j + 2] == r')'):
                                rate = 1.
                            else:
                                step = 2
                                string = ''
                                while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                        (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                    string = string + matname[n][i][j + step]
                                    step = step + 1
                                rate = float(string)
                        elif ((ord(matname[n][i][j + 1]) > 64) & (ord(matname[n][i][j + 1]) < 91)) or (
                                matname[n][i][j + 1] == r' ') or (matname[n][i][j + 1] == r'(') or (
                                matname[n][i][j + 1] == r')'):
                            rate = 1.
                        else:
                            step = 1
                            string = ''
                            while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                    (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                string = string + matname[n][i][j + step]
                                step = step + 1
                            rate = float(string)
                    rep = 0
                    for k in range(len(elements)):
                        if (elements[k] == ele):
                            rep = 1
                            ratio[k] = ratio[k] + rate
                    if (rep == 0):
                        elements.append(ele)
                        ratio.append(rate)
            composition[n].append(ratio)
            elist[n].append(elements)


            ea, ip, en, rc, ve, pl = [], [], [], [], [], []
            tv = 0.
            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        tv = tv + float(ratio[j]) * float(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        pl.append(polar[k])
                        ve.append(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele1)):
                    if (elements[j] == ele1[k]):
                        ip.append(eleip1[k])
                        ea.append(elena1[k])
                        en.append(elene1[k])
                        rc.append(rcov1[k])

            for k in range(len(ele1)):
                if sqatom[n][i] == ele1[k]:
                    ipsq = eleip1[k]
                    easq = elena1[k]
                    ensq = elene1[k]
                    rcsq = rcov1[k]

            for k in range(len(ele2)):
                if sqatom[n][i] == ele2[k]:
                    plsq = polar[k]
                    vesq = valence[k]
                    fccsq = fcc[k]

            datapoint = [min(en), ensq, max(ve), min(ve), vesq, tv, dsq[n][i], dv[n][i],
                         fccsq, max(ea), min(ea), easq]
            Xtot[n].append(datapoint)
            Ytot[n].append(labels[i])

        print(len(Xtot[n]), len(Xtot[n][0]))


    



    X1 = [[] for i in range(len(group))]
    X0 = [[] for i in range(len(group))]
    for n in range(len(group)):
        count1 = 0
        count0 = 0
        for i in range(len(Ytot[n])):
            if Ytot[n][i] == 1:
                X1[n].append(Xtot[n][i])
                count1 = count1 + 1
            else:
                X0[n].append(Xtot[n][i])
                count0 = count0 + 1
        print(n, count1, count0)

    for j in range(len(X1[0][0])):
        buffer=[]
        for n in range(2):
            for i in range(len(X1[n])):
                buffer.append(X1[n][i][j])
            for i in range(len(X0[n])):
                buffer.append(X0[n][i][j])
        len1=np.max(buffer)-np.min(buffer)
        min1=np.min(buffer)
        print(j,'len',len1)
        for n in range(2):
            for i in range(len(X1[n])):
                X1[n][i][j]-=min1
                X1[n][i][j]/=len1
            for i in range(len(X0[n])):
                X0[n][i][j]-=min1
                X0[n][i][j]/=len1

    return X1, X0