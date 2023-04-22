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

def gradientDown():
    dictGradDown,k = {'x1':[x0[0]],'x2':[x0[1]], 'f':[func(x0[0],x0[1])]},0
    norm = lambda x,y:int(x)**2+int(y)**2
    while (k)<M:
        print(f'''
        Метод градиента с постоянным шагом итерация(k = {k})
        точка ({dictGradDown['x1'][k]},{dictGradDown['x1'][k]})
        градиент функции в точке {(dfxk:=derivF(dictGradDown['x1'][k],dictGradDown['x2'][k]))}
        Проверка услови оконччания {(gr:=int(norm(dfxk[0],dfxk[1]))**0.5)} {'< '+str(e[0]) if gr<e[0] else '≥ '+str(e[0])+' Не'} удовлетворяет
        ''')
        if gr<e[1]:break
        tk,fxx,fxk = getNumber02('величину шага tk ', 'float'),1,0
        while not (fxx < fxk):
            print(f'''
        Текущее значение tk {tk}
        x(k+1) = {(xk1:=[dictGradDown['x1'][k]-tk*int(dfxk[0]),dictGradDown['x2'][k]-tk*int(dfxk[1])])}''')
            tk/=2
            dictGradDown['x1'] = dictGradDown['x1'][:k+1]+[xk1[0]]
            dictGradDown['x2'] = dictGradDown['x2'][:k+1]+[xk1[1]]
            print(
                f'Шаг 8 {(fxx:=func(dictGradDown["x1"][-1],dictGradDown["x2"][-1]))} {"< "+str(fxk:=(fxk:=func(dictGradDown["x1"][k],dictGradDown["x2"][k]))) if fxx<fxk else "≥ "+str(fxk)+" Не"} удовлетворяет')
        print(f'''
        Условие 1: {(usl1:=int(norm(dictGradDown["x1"][k+1]-dictGradDown["x1"][k],dictGradDown["x2"][k+1]-dictGradDown["x2"][k])) ** 0.5)} {'< '+str(e[1]) if usl1<e[1] else '≥ '+str(e[1])+' Не'} удовлетворяет
        Условие 2: {(usl2:=abs(
                func(dictGradDown['x1'][k + 1],dictGradDown['x2'][k + 1]) - fxk))} {'< '+str(e[1]) if usl2<e[1] else '≥ '+str(e[1])+' Не'} удовлетворяет
        ''')
        if (usl1 < e[1] and usl2 < e[1]):break
        k+=1
    print(f'Ответ k = {len(dictGradDown["x1"])-1}; x {(dictGradDown["x1"][-1],dictGradDown["x1"][-1])} {"в связи с ограничением M" if k>=M else ""}')

def fastgradientDown():
    pass


#x1, x1x2, x2 = map(lambda x: getNumber02(f'коэффицент для {x} = ', 'int'), ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])
#x0 = list(map(lambda x: getNumber02(f'начальную точку {x} ', 'float'), ["x", "y"]))
#e = list(map(lambda x: getNumber02(f'точность {x} ', 'float'), ["e1", "e2"]))
#M = getNumber02('предел числа операций', 'int')
x1,x1x2,x2,x0,e,M=1,0.5,5,(0,0.5),(0.15,0.2),10
z = lambda x, y: "+" + str(x) + y if x > 0 else (str(x) + y if x < 0 else "")
func = lambda x,y: x1 * (int(x) ** 2) + x1x2 *x*y + (int(y)**2) * x2
derivF = lambda x,y: [2*x1 * x + x2*y, x1*x + y * x2*2]
print(f'Ваша функция f(x) = {("".join(list(map(z, [x1, x1x2, x2], ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])))).lstrip("+").lstrip("1")}')
gradientDown()
