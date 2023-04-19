import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graphic(dict, name):
    for k in ['y','z','x']:
        while len(dict[k]) < len(dict['a']):
            addtoDict(dict,k,dict[k][-1])
    for k in range(len(dict["a"])): map(lambda x: addtoDict(dict,'f'+x,func(dict[x][k])), ['a','b','y','z','x'])
    df = pd.DataFrame(dict)
    for k in ['a', 'b', 'y', 'z', 'x']:df[str('f'+k)] = list(map(func, dict[k]))
    print(df)
    x = np.linspace(-2, 2, 200)
    graf = sns.FacetGrid(data=df, col='a')
    for elem in range(len(df['a'])): graf.axes[0,elem].plot(x, x3 * (x ** 3) + x2 * (x ** 2) + x1 * x + c)
    plt.xlim(interval)
    plt.ylim(min(df['fx']),max(df['fa'][0],df['fb'][0]))
    graf.axes[0, len(df['a']) - 1].plot(ylim =(-6,-5))
    graf.fig.set_size_inches(15,15)
    graf.map(plt.scatter, 'a', 'fa', color='red', alpha=0.4)
    graf.map(plt.scatter, 'b', 'fb', color='red', alpha=0.4)
    graf.map(plt.scatter, 'y', 'fy', color='green', alpha=0.4)
    graf.map(plt.scatter, 'z', 'fz', color='blue', alpha=0.4)
    graf.map(plt.scatter, 'x', 'fx', color='orange', alpha=0.4)
    graf.set(title=name)
    plt.show()

def getNumber02(x, type):  # проверка на ввод числа
    while True:
        if type == 'int':
            try:
                getNumber = int(input('Введите ' + x))
                return getNumber
            except ValueError:
                print('Введите запись типа: ' + str(type))
        elif type == 'float':
            try:
                getNumber = float(input('Введите ' + x))
                return getNumber
            except ValueError:
                print('Введите запись типа: ' + str(type))


def addtoDict(dictionary, name, param):
    dictionary[name] = dictionary.get(name, []) + [param]
    return dictionary[name][-1]


def methodHalfDivision(interval, accuracy):
    dictHalfDiv = {"a": [interval[0]], "b": [interval[1]], "x": [sum(interval) / 2],
                   "l": [abs(interval[0] - interval[1])], "y": [], "z": []}
    while dictHalfDiv['l'][(k := (dictHalfDiv["x"].__len__() - 1))] > accuracy:
        print(f'''
    Метод половинного деления: Итерация {k} 
        Отрезок [{dictHalfDiv['a'][k]},{dictHalfDiv['b'][k]}]
        Средняя точка {dictHalfDiv['x'][k]}
        Длина интервала {dictHalfDiv['l'][k]}
        Значение функции в средней точке {(fxk := func(dictHalfDiv["x"][k]))}
        y = {addtoDict(dictHalfDiv, "y",
                       (dictHalfDiv["a"][k] + dictHalfDiv["l"][k] / 4))}; z = {addtoDict(dictHalfDiv, "z",
                                                                                         (dictHalfDiv["b"][k] - dictHalfDiv["l"][k] / 4))}
        Значение f(у) = {(fyk := func(dictHalfDiv["y"][k]))}; f(z) = {(fzk := (func(dictHalfDiv["z"][k])))}''')
        if fyk < fxk:
            for x, y in ['b', 'x'], ['a', 'a'], ['x', 'y']: addtoDict(dictHalfDiv, x, dictHalfDiv[y][k])
        else:
            if fzk < fxk:
                for x, y in ['a', 'x'], ['b', 'b'], ['x', 'z']: addtoDict(dictHalfDiv, x, dictHalfDiv[y][k])
            else:
                for x, y in ['a', 'y'], ['b', 'z'], ['x', 'x']: addtoDict(dictHalfDiv, x, dictHalfDiv[y][k])
        print(
            f'\tновый отрезок от {dictHalfDiv["a"][k + 1]} до {dictHalfDiv["b"][k + 1]} с центром в {dictHalfDiv["x"][k + 1]}')
        dictHalfDiv['l'].append(abs(dictHalfDiv["a"][k + 1] - dictHalfDiv["b"][k + 1]))
    print(
        f'В качестве приближения  точка x: {dictHalfDiv["x"][k-1]} f(x) = {func(dictHalfDiv["x"][k-1])}')
    return dictHalfDiv


def methodGoldenSecion(interval, accuracy):
    dictGoldSec = {"a": [interval[0]], "b": [interval[1]],
                   "y": [interval[0] + ((3 - 5 ** (0.5)) / 2) * (interval[1] - interval[0])], "z": [], "x": []}
    addtoDict(dictGoldSec, "z", (sum(interval) - dictGoldSec["y"][0]))
    while abs(dictGoldSec['a'][(k := (dictGoldSec["a"].__len__() - 1))] - dictGoldSec['b'][k]) > accuracy:
        print(f'''
    Метод золотого сечения: Итерация {k} 
        Отрезок [{dictGoldSec['a'][k]},{dictGoldSec['b'][k]}]
        y = {dictGoldSec["y"][k]}; z = {dictGoldSec["z"][k]}
        Значение f(у) = {(fyk := func(dictGoldSec["y"][k]))}; f(z) = {(fzk := func(dictGoldSec["z"][k]))}''')
        if fyk > fzk:
            for x, y in ['a', 'y'], ['b', 'b'], ['y', 'z']: addtoDict(dictGoldSec, x, dictGoldSec[y][k])
            dictGoldSec["z"].append(dictGoldSec['a'][k + 1] + dictGoldSec['b'][k + 1] - dictGoldSec['z'][k])
        else:
            for x, y in ['a', 'a'], ['b', 'z'], ['z', 'y']: addtoDict(dictGoldSec, x, dictGoldSec[y][k])
            dictGoldSec["y"].append(dictGoldSec['a'][k + 1] + dictGoldSec['b'][k + 1] - dictGoldSec['y'][k])
        print(
            f'\tновый отрезок от {dictGoldSec["a"][k + 1]} до {dictGoldSec["b"][k + 1]} расстояние {(l := dictGoldSec["a"][k + 1] - dictGoldSec["b"][k + 1])}{" > " if abs(l) > accuracy else " <= "}{accuracy}')
    print(
        f'В качестве приближения  точка x: {addtoDict(dictGoldSec, "x", (abs(dictGoldSec["a"][k + 1] - dictGoldSec["b"][k + 1]) / 2))} f(x) = {func(dictGoldSec["x"][-1])}')
    return dictGoldSec


def methodFibonachi(interval, accuracy):
    massFibonachi = fibonachi(abs(interval[0] - interval[1]) / accuracy)
    hardf = lambda a, b, up, down: int(dictFibonachi['a'][a] + (massFibonachi[up] / massFibonachi[down]) * (
            dictFibonachi['b'][b] - dictFibonachi['a'][a]))
    dictFibonachi = {"a": [interval[0]], "b": [interval[1]], "y": [hardf(0, 0, -3, -1)], "z": [hardf(0, 0, -2, -1)],
                     "x": []}
    while (k := len(dictFibonachi['a'])) != len(massFibonachi) - 3:
        print(f'''
    Метод Фибоначи: Итерация {k}
        y = {dictFibonachi["y"][k]}; z = {dictFibonachi["z"][k]}
        Значение f(у) = {(fyk := func(dictFibonachi["y"][k]))}; f(z) = {(fzk := func(dictFibonachi["z"][k]))}''')
        if fyk > fzk:
            for x, y in ['a', 'y'], ['b', 'b'], ['y', 'z']: addtoDict(dictFibonachi, x, dictFibonachi[y][k])
            dictFibonachi["z"].append(hardf(k + 1, k + 1, -3 - k, -2 - k))
        else:
            for x, y in ['a', 'a'], ['b', 'z'], ['z', 'y']: addtoDict(dictFibonachi, x, dictFibonachi[y][k])
            dictFibonachi["y"].append(hardf(k + 1, k + 1, -4 - k, -2 - k))
        print(f'''\tновый отрезок от {dictFibonachi["a"][k + 1]} до {dictFibonachi["b"][k + 1]}
        Условие: {k} {'≠' if k != len(massFibonachi) else '='} {len(massFibonachi)}-{3}''')
    for x, y, z in ['y', 'z', 0], ['z', 'z', accuracy]: dictFibonachi[x].append(dictFibonachi[y][k] + z)
    if func(dictFibonachi['y'][k + 1]) > func(dictFibonachi['z'][k + 1]):
        for x, y, z in ['a', 'y', 1], ['b', 'b', 0]: addtoDict(dictFibonachi, x, dictFibonachi[y][k + z])
    else:
        for x, y, z in ['a', 'a', 0], ['b', 'z', 1]: addtoDict(dictFibonachi, x, dictFibonachi[y][k + z])
    print(
        f'В качестве приближения точка x = {addtoDict(dictFibonachi, "x", (dictFibonachi["a"][k + 1] + dictFibonachi["b"][k + 1]) / 2)}; f(x) = {func(dictFibonachi["x"][-1])}')
    return dictFibonachi


def fibonachi(number):
    fib = [1, 1]
    while number < fib[-1]:
        fib.append(fib[-1] + fib[-2])
    return fib


'''
Вариант 19
f(x) = x^3-12x-5
L_0 = [1,3]
E = 0,2
'''

#x3, x2, x1, c = map(lambda x: getNumber02(f'коэффицент для {x} = ', 'int'), ["x^3", "x^2", "x", "c"])
#interval = list(map(lambda x: getNumber02(f'{x} интервала ', 'int'), ["начало", "конец"]))
#e = getNumber02('точность E ', 'float')
z = lambda x, y: "+" + str(x) + y if x > 0 else ("-" + str(x) + y if x < 0 else "")
func = lambda x: x3 * (x ** 3) + x2 * (x ** 2) + x1 * x + c
x3,x2,x1,c,interval,e = 2,9,0,-6,[-1,1],0.2
print(f'Ваша функция f(x) = {("".join(list(map(z, [x3, x2, x1], ["x^3", "x^2", "x"])))+str(c)).lstrip("+")}')

'''
все элементы словаря должны быть одинаковой длинны
'''
#p = pd.DataFrame(methodHalfDivision(fX, interval, e))
#print(f'type: {type(p["a"][0])}')
graphic(methodHalfDivision(interval, e), 'Метод половинного деления')
#methodGoldenSecion(fX, interval, e)
#methodFibonachi(fX, interval, e)
