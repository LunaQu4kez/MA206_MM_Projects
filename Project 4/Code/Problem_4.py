from typing import Dict
import pandas as pd
from pandas import DataFrame
df_2 = pd.read_excel('Appendix_2.xlsx')
df_3 = pd.read_excel('Appendix_3.xlsx')
df_3.head()

def getEmptyDictPair(data):
    dic_pair = {}
    for index, row in data.iterrows():
        start = row[0]
        end = row[1]
        fix = row[2]
        pcs = row[3]
        dic_pair[(start, end)] = (fix, pcs)
    return dic_pair

day_23 = df_2[df_2.iloc[:, 0] == '2023-04-23']
day_23.head()

dic_nx = {}

lst_pair = [('A', 'C'), ('A', 'T'), ('C', 'L'), ('L', 'T'), ('T', 'E'), ('L', 'N'),
 ('N', 'E'), ('N', 'Q'), ('Q', 'P'), ('E', 'P'), ('E', 'W'), ('W', 'U'),
 ('P', 'U'), ('W', 'G'), ('G', 'U'), ('P', 'V'), ('V', 'F'), ('U', 'F'), 
 ('G', 'O'), ('O', 'F'), ('F', 'R'), ('R', 'K'), ('V', 'K'), ('R', 'X'), ('O', 'X'), ('O', 'M'), ('K', 'J'), ('J', 'X'), ('X', 'I'), ('M', 'I'), ('M', 'D'), ('D', 'S'), ('D', 'Y'), ('S', 'Y'), ('S', 'I'), ('I', 'B'), ('B', 'J'), ('B', 'H'), ('H', 'J')]

for pair in lst_pair:
    dic_nx[pair] = 0

def calWeight(fix, pcs, real):
    return fix * (1 + (1.0*real/pcs)**3)

# 把某天的实际货运数据转化成日期的 dict

def getDictDay(data_day: DataFrame, pairs) -> Dict:
    dic_real = {}
    
    for pair in pairs:
        dic_real[pair] = 0
    
    for index, row in data_day.iterrows():
        start = row[1]
        end = row[2]
        real = row[3]
        dic_real[(start, end)] = real
        
        
    return dic_real

def getNx(day, start, end):
    dic_nx = {}

    pairs = [
        ('A', 'C'), ('A', 'T'), ('C', 'L'), ('L', 'T'), ('T', 'E'), ('L', 'N'),
        ('N', 'E'), ('N', 'Q'), ('Q', 'P'), ('E', 'P'), ('E', 'W'), ('W', 'U'),
        ('P', 'U'), ('W', 'G'), ('G', 'U'), ('P', 'V'), ('V', 'F'), ('U', 'F'), 
        ('G', 'O'), ('O', 'F'), ('F', 'R'), ('R', 'K'), ('V', 'K'), ('R', 'X'), 
        ('O', 'X'), ('O', 'M'), ('K', 'J'), ('J', 'X'), ('X', 'I'), ('M', 'I'), 
        ('M', 'D'), ('D', 'S'), ('D', 'Y'), ('S', 'Y'), ('S', 'I'), ('I', 'B'), 
        ('B', 'J'), ('B', 'H'), ('H', 'J')]

    for pair in pairs:
        dic_nx[pair] = 0
            
    row = df_3[df_3['起点 (Start)'] == start][df_3['终点 (End)'] == end]
    if(row.empty):
        return dic_nx
    fix = row.iloc[0, 2]
    pcs = row.iloc[0, 3]
    
    dic_day = getDictDay(day, pairs)
    
    for key, value in dic_nx.items():
        real = dic_day[key]
        dic_nx[key] = fix * (1 + (1.0*real/pcs)**3)

    return dic_nx
    
from networkx.classes.function import path_weight
def calculateTheDay(day, pairs):
    res = 0
    dic_real = getDictDay(day, pairs)
    
    for key, value in dic_real.items():
        dic_nx = getNx(day, key[0], key[1])
        network = nx.Graph()
        for pair, weight in dic_nx.items():
            network.add_edge(pair[0], pair[1], weight=weight) 

        if(nx.shortest_path_length(network, key[0], key[1]) > 5):
            continue
        else:
            res += path_weight(network, nx.shortest_path(network, key[0], key[1], weight='weight'), weight='weight')
    
    return res
            
        
result = calculateTheDay(df_2[df_2.iloc[:, 0] == '2023-04-27'], pairs)
result