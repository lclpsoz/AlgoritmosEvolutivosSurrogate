import os
import numpy as np
from statistics import stdev
from pprint import pprint
from tabulate import tabulate
from scipy.stats import wilcoxon
import json
from copy import deepcopy
from copy import copy

def avr(lst):
    return sum(lst)/len(lst)

def mark_best(data, problems):
    data_marked = deepcopy(data)
    for key in data:
        for prob in problems:
            data_marked[key][prob] = 0

    for prob in problems:
        print("PROBLEM =", prob)
        base_avr = 10*100
        for key in data:
            avr_now = avr(data[key][prob])
            if avr_now < base_avr:
                base_avr = avr_now
                base = key
        print(base)
        data_marked[base][prob] = 1
        for target in data:
            if target != base:
                stat, p = wilcoxon(data[base][prob], data[target][prob])
                if p > 0.05:
                    if data_marked[target][prob] == 0:
                        data_marked[target][prob] = 2
                    print('---------------------')
                    print(avr(data[target][prob]))
                    print(avr(data[base][prob]))
                    print('----------------------')

    return data_marked

def proc(folder_name : str):
    if folder_name[-1] != '/':
        folder_name += '/'
    data = {}
    for filename in os.listdir(folder_name):
        if(os.path.isfile(folder_name + filename)):
            with open(folder_name + filename, 'r') as fp:
                lst = filename.split('_')
                surrogate = lst[5].split('-')[0]
                if 'LSTM' in surrogate or 'RNN' in surrogate:
                    prop = lst[6].split('=')[1]
                    amntEpochs = lst[7].split('=')[1]
                    ts = lst[8].split('=')[1]
                    surrogate_type = lst[9]
                    if amntEpochs == '0':
                        surrogate = surrogate + '\_NO-TRAIN'
                    else:
                        surrogate = surrogate + '\_p=' +\
                                    prop + '\_ts=' +\
                                    ts + '\_' +\
                                    surrogate_type
                else:
                    if 'NO' in surrogate:
                        surrogate = 'Sem surrogate'
                    else:
                        surrogate += '_' + lst[6]
                mp = {}
                for row in fp.read().split('\n'):
                    if(len(row) > 1):
                        row = row.split(';')
                        if not row[0] in mp:
                            mp[row[0]] = []
                        mp[row[0]].append(float(row[1]))
                try:
                    data[surrogate].update(mp)
                except:
                    data[surrogate] = mp
    # pprint(data)
    with open('data.json', 'w') as fp:
        json.dump(data, fp)
    return data

def generate_latex_table(data, lst_problems : list, marked_data):
    table = []
    meta_models = sorted(data.keys())
    table_csv = []
    for meta_model in meta_models:
        print(meta_model)
        q = 0
        for p in lst_problems:
            if(marked_data[meta_model][p] > 0):
                print(meta_model, marked_data[meta_model][p])
            if marked_data[meta_model][p] > 0:
                q+=1
            if marked_data[meta_model][p] == 1:
                q = 10
        if ('LSTM' in meta_model or 'RNN' in meta_model) and q <= 1: # and not 'NO-TRAIN' in meta_model:
            continue
        meta_model_id = meta_model
        meta_model = meta_model.split('\_')
        if len(meta_model) > 2:
            lst_avr = [ meta_model[0],
                        float(meta_model[1].split('-')[1])/100,
                        meta_model[2].split('=')[1],
                        meta_model[3]
            ]
            lst_stdev = ['', '', '', '']
        elif 'RAMDOM' in meta_model_id:
            meta_model = meta_model_id.split('_')
            lst_avr = [ '\\textit{Random Forest}',
                        '-',
                        '-',
                        meta_model[1]
            ]
            lst_stdev = ['', '', '', '']
        elif 'TREE' in meta_model_id:
            meta_model = meta_model_id.split('_')
            lst_avr = [ 'Árvore de Decisão',
                        '-',
                        '-',
                        meta_model[1]
            ]
            lst_stdev = ['', '', '', '']
        else:
            lst_avr = [ meta_model[0],
                        '-',
                        '-',
                        '-'
            ]
            lst_stdev = ['', '', '', '']
            l_avr = copy(lst_avr)
            l_stdev = ['Meta-model']

        for problem in lst_problems:
            vals = data[meta_model_id][problem]
            avr = sum(vals)/len(vals)
            # print(meta_model, surrogate, problem, avr, stdev(vals))
            str_avr = "%.5f $\pm$" % (avr)
            str_stdev = "%.5f" % (stdev(vals))
            if marked_data[meta_model_id][problem] > 0:
                str_avr = '\\textbf{' + str_avr + '}'
                str_stdev = '\\textbf{' + str_stdev + '}'
            lst_avr.append(str_avr)
            lst_stdev.append(str_stdev)

            # l_avr.append(avr)
            # l_stdev.append(stdev(vals))
        table.append(lst_avr)
        table.append(lst_stdev)
        # table_csv.append(l_avr)
        # table_csv.append(l_stdev)
    # for row in table_csv:
    #     print(row[0], end=',')
    #     for i in range(1, len(row)):
    #         print(row[i], end=',')
    #     print('')

    return table

def add_midrule_to_latex_str(latex_str : str):
    latex_str = latex_str.split('\n')
    latex_str[1] = '\\toprule'
    latex_str[3] = '\\midrule'
    latex_str[-2] = '\\bottomrule'
    for i in range(0, len(latex_str)-1):
        # print(latex_str[i][-1], latex_str[i+1][0])
        if latex_str[i][-1]  == '\\' and not latex_str[i+1].startswith('  ') and latex_str[i+1][0] != '\\':
            latex_str[i] += ' \\hline'
    return '\n'.join(latex_str)

lst_problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'WFG1', 'WFG2', 'WFG3']
lst_problems = []
# for i in range(1, 10):
#     lst_problems.append('WFG' + str(i))
for i in range(1, 4):
    lst_problems.append('DTLZ' + str(i))
for i in range(1, 4):
    lst_problems.append('WFG' + str(i))

data = proc('data_st_3')

marked_data = mark_best(data, lst_problems)

# pprint(data)
table_latex = generate_latex_table(data, lst_problems, marked_data)
# print(table_latex)
# table[0][0] = 'NSGAII45'
# table[2][0] = 'NSGAII45SBmyoAD'

print(add_midrule_to_latex_str(tabulate(table_latex, headers=['Meta-modelo', '$x$', '$ts$', 'Método'] + lst_problems, tablefmt='latex_raw')))