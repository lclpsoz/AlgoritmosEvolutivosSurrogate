import os
from statistics import stdev
from pprint import pprint
from tabulate import tabulate
from scipy.stats import wilcoxon
import json

def add_new_lines():
    for filename in os.listdir('files_igd'):
        if(os.path.isfile('files_igd/' + filename)):
            with open('files_igd/' + filename, 'r') as fp:
                text = fp.read()
            if(('DTLZ' in text and not '\nDTLZ2' in text) or
                ('WFG' in text and not '\nWFG2' in text)):
                print('here')
                with open('files_igd/' + filename, 'w') as fp:
                    text = text.replace('DTLZ', '\nDTLZ')
                    text = text.replace('WFG', '\nWFG')
                    fp.write(text)

# add_new_lines()

def proc():
    opt_all = [2, 4, 5, 6, 7, 8, 9]
    opt = [2, 4, 5, 6]

    data = {}
    for filename in os.listdir('files_igd'):
        if(os.path.isfile('files_igd/' + filename)):
            if(True):
                with open('files_igd/' + filename, 'r') as fp:
                    lst = filename.split('_')
                    surrogate = lst[5]
                    surrogate_type = lst[6]
                    mp = {}
                    for row in fp.read().split('\n'):
                        if(len(row) > 1):
                            row = row.split(';')
                            if not row[0] in mp:
                                mp[row[0]] = []
                            mp[row[0]].append(float(row[1]))
                    if surrogate in data:
                        try:
                            data[surrogate][surrogate_type].update(mp)
                        except:
                            data[surrogate][surrogate_type] = mp
                    else:
                        data[surrogate] = {
                            'Online' : {},
                            'Batch' : {}
                        }
                        data[surrogate][surrogate_type] = mp
    pprint(data)
    with open('data.json', 'w') as fp:
        json.dump(data, fp)
    return data

data = proc()
lst_problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'WFG1', 'WFG2', 'WFG3']
table = []
dtlz1 = []
for meta_model in data:
    for surrogate in data[meta_model]:
        lst_avr = [meta_model + '\_' + surrogate]
        lst_stdev = ['']
        for problem in lst_problems:
            vals = data[meta_model][surrogate][problem]
            # print(vals)
            avr = sum(vals)/len(vals)
            # print([x-avr for x in vals])
            stat, p = wilcoxon([x-avr for x in vals])
            # print(stat, p)
            # if(p > 0.05):
            #     print('-------------')
            print(meta_model, surrogate, problem, avr, stdev(vals))

            lst_avr.append("%.5f $\pm$" % (avr))
            lst_stdev.append("%.5f" % (stdev(vals)))
        dtlz1.append((meta_model, surrogate, data[meta_model][surrogate]['DTLZ3']))
        table.append(lst_avr)
        table.append(lst_stdev)

# print(len(dtlz1))
# for i in range(len(dtlz1)):
#     d1 = dtlz1[i]
#     for j in range(i+1, len(dtlz1)):
#         d2 = dtlz1[j]
#         stat, p = wilcoxon(d1[2], d2[2])
#         if p > 0.05:
#             print(d1[0], d1[1], d2[0], d2[1], p)
print(tabulate(table, headers=['Modelo\_Surrogate'] + lst_problems, tablefmt='latex_raw'))