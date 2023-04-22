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


def addtoDict(dictionary, name, param,place=1):
    dictionary[name] = dictionary.get(name, []) + [param]
    return dictionary[name][-1]

def gradientDown(M):
    dictGradDown = {'x1':[x0[0]],'x2':[x0[1]], 'f':[func(x0)]}
    norm = lambda x:dictGradDown['x1'][x]**2+dictGradDown['x2'][x]**2
    while (k:=0)<M:
        print(f'''
        Метод градиента с постоянным шагом итерация(k = {k})
        точка ({dictGradDown['x1'][k]},{dictGradDown['x1'][k]})
        градиент функции в точке {(dfxk:=derivF(dictGradDown['x1'][k],dictGradDown['x2'][k]))}
        Проверка услови оконччания {(gr:=(norm(k))**0,5)} {'>' if gr>e[0] else '=' if gr==e[0] else '<'} 
        ''')
        if gr<e[1]:break
        tk = getNumber02('величину шага tk ', 'float')
        while not (func(dictGradDown['x1'][-1],dictGradDown['x2'][-1]) < func(dictGradDown['x1'][k],dictGradDown['x2'][k])):
            print(f'''
        Текущее значение tk {tk}
        x(k+1) = {(xk1:=(dictGradDown['x1'][k]-tk*dfxk[0],dictGradDown['x2'][k]-tk*dfxk[1]))}''')
            tk/=2;dictGradDown['x1'] = [dictGradDown['x1'][:k+1]+[xk1[0]]];dictGradDown['x2'] = [dictGradDown['x2'][:k+1]+[xk1[1]]]
        print(f'''
        Условие 1: {(usl1:=norm(k + 1) ** 0, 5)} {'< '+str(e[1]) if usl1<e[1] else '≥ '+str(e[1])+' Не '} удовлетворяет
        Условие 2: {(usl2:=abs(
                func(dictGradDown['x1'][k + 1], dictGradDown['x2'][k + 1]) - func(dictGradDown['x1'][k + 1],
                                                                                  dictGradDown['x2'][k + 1])) < e[1])} {'< '+str(e[1]) if usl2<e[1] else '≥ '+str(e[1])+' Не '} удовлетворяет
        ''')
        if (usl1 < e[1] and usl2 < e[1]):break
        k+=1
    print(f'Ответ x_{len(dictGradDown["x"])} {(dictGradDown[x1][-1],dictGradDown[x1][-1])}{"в связи с ограничением M" if k>=M else ""}')

def fastgradientDown():
    pass



x1, x1x2, x2 = map(lambda x: getNumber02(f'коэффицент для {x} = ', 'int'), ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])
x0 = list(map(lambda x: getNumber02(f'начальную точку {x} ', 'float'), ["x", "y"]))
e = list(map(lambda x: getNumber02(f'точность {x} ', 'float'), ["e1", "e2"]))
M = getNumber02('предел числа операций', 'int')
z = lambda x, y: "+" + str(x) + y if x > 1 else (str(x) + y if x < 0 else "")
func = lambda x,y: x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2
derivF = lambda x,y: 2* (x1 * x + x2*y, x1*x + y * x2*2)
#x3,x2,x1,c,interval,e = 1,0.5,5,-6,[0,0.5],10
print(f'Ваша функция f(x) = {("".join(list(map(z, [x1, x1x2, x2], ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])))).lstrip("+")}')
