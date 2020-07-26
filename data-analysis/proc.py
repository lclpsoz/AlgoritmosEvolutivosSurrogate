import os
from statistics import stdev
from pprint import pprint
from tabulate import tabulate
from scipy.stats import wilcoxon
import json

def proc(folder_name : str):
    if folder_name[-1] != '/':
        folder_name += '/'
    data = {}
    for filename in os.listdir(folder_name):
        if(os.path.isfile(folder_name + filename)):
            with open(folder_name + filename, 'r') as fp:
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
                    data[surrogate] = {}
                    data[surrogate][surrogate_type] = mp
    # pprint(data)
    with open('data.json', 'w') as fp:
        json.dump(data, fp)
    return data

def generate_latex_table(data, lst_problems : list):
    table = []
    dtlz1 = []
    for meta_model in data:
        for surrogate in data[meta_model]:
            lst_avr = [meta_model + '\_' + surrogate]
            lst_stdev = ['']
            for problem in lst_problems:
                vals = data[meta_model][surrogate][problem]
                avr = sum(vals)/len(vals)
                # print(meta_model, surrogate, problem, avr, stdev(vals))

                lst_avr.append("%.5f $\pm$" % (avr))
                lst_stdev.append("%.5f" % (stdev(vals)))
            table.append(lst_avr)
            table.append(lst_stdev)
    
    return table

def add_midrule_to_latex_str(latex_str : str):
    latex_str = latex_str.split('\n')
    latex_str[1] = '\\toprule'
    latex_str[3] = '\\midrule'
    latex_str[-2] = '\\bottomrule'
    for i in range(0, len(latex_str)-1):
        print(latex_str[i][-1], latex_str[i+1][0])
        if latex_str[i][-1]  == '\\' and not latex_str[i+1].startswith('  ') and latex_str[i+1][0] != '\\':
            latex_str[i] += ' \\hline'
    return '\n'.join(latex_str)

# lst_problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'WFG1', 'WFG2', 'WFG3']
lst_problems = []
# for i in range(1, 10):
#     lst_problems.append('WFG' + str(i))
for i in range(1, 8):
    lst_problems.append('DTLZ' + str(i))

table = generate_latex_table(proc('files_igd'), lst_problems)
print(table)
table[0][0] = 'NSGAII45'
table[2][0] = 'NSGAII45SBmyoAD'

print(add_midrule_to_latex_str(tabulate(table, headers=[''] + lst_problems, tablefmt='latex_raw')))