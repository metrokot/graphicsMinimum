import numpy as np
import matplotlib.pyplot as plt


def graphic(dictionary, name):#построить окружность(?)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    plt.contour(x, y, func(x,y), levels=[i for i in reversed(dictionary['f'])])
    plt.title(name)
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
def extremumMin(x,y,a,b,c,situation=0):
    exp1 = lambda : 2*a*b**2*x**2+4*a*b*c*x*y+b**3*x*y+2*b**2*c*y**2
    exp2 = lambda: b*x**2+4*b*c*x*y+4*c**2*y**2
    exp3 = lambda: -2*b**2*c*x**2-8*b*c**2*x*y-8*c**3*y**2
    v1 = lambda: 8*a**3*x**2+8*a**2*b*x*y+2*a*b**2*y**2
    v2 = lambda: 2*a*b**2*x**2+4*a*b*c*x*y+b**3*x*y+2*b**2*c*y**2
    v3 = lambda: 4*a**2*x**2+4*a*b*x*y+b**2*y**2
    if(situation==1):
        t1 = (v3()+(exp2()*v2())/exp3())/(v1()+(exp1()*v2())/exp3())
        t2 = (t1*exp1()-exp2())/exp3()
        return [t1,t2]
    proverca1 = lambda: -4*a**2*x**2+4*a*b*x*y+b**2*x**2+4*b*c*x*y+b**2*y**2+4*c*y**2
    proverca2 = lambda: v1()+4*a*b**2*x**2+8*a*b*c*x*y+2*b**3*x*y+4*b**2*c*y**2-4*c*y**2-exp3()
    return [proverca1()/proverca2()]*2


def gradientDown():
    dictGradDown,k = {'x1':[x0[0]],'x2':[x0[1]], 'f':[]},0
    while k<M:
        print(f'''
        Метод градиента с постоянным шагом итерация (k = {k})
        точка ({dictGradDown['x1'][k]},{dictGradDown['x2'][k]})
        градиент функции в точке {(dfxk:=derivF(dictGradDown['x1'][k],dictGradDown['x2'][k]))}
        {norm(dfxk[0],dfxk[1])}
        Проверка условия оконччания {(gr:=float(norm(dfxk[0],dfxk[1]))**0.5)} {'< '+str(e[0]) if gr<e[0] else '≥ '+str(e[0])+' Не'} удовлетворяет
        ''')
        if gr<e[1]:break
        tk,fxk1,fxk = getNumber02('величину шага tk ', 'float'),1,0
        while not (fxk1 < fxk):
            print(f'''
        Текущее значение tk {tk}
        x(k+1) = {(xk1:=[dictGradDown['x1'][k]-tk*dfxk[0],dictGradDown['x2'][k]-tk*dfxk[1]])}''')
            tk/=2
            dictGradDown['x1'] = dictGradDown['x1'][:k+1]+[xk1[0]]
            dictGradDown['x2'] = dictGradDown['x2'][:k+1]+[xk1[1]]
            print(
                f'Шаг 8 {(fxk1:=func(dictGradDown["x1"][-1],dictGradDown["x2"][-1]))} {"< " if fxk1<(fxk:=func(dictGradDown["x1"][k],dictGradDown["x2"][k])) else "≥ "} {fxk} {"Не " if fxk1>=fxk else ""}удовлетворяет')
        print(f'''
        Условие 1: {(usl1:=float(norm(dictGradDown["x1"][k+1]-dictGradDown["x1"][k],dictGradDown["x2"][k+1]-dictGradDown["x2"][k])) ** 0.5)} {'< '+str(e[1]) if usl1<e[1] else '≥ '+str(e[1])+' Не'} удовлетворяет
        Условие 2: {(usl2:=abs(
                fxk1 - addtoDict(dictGradDown,'f',fxk)))} {'< '+str(e[1]) if usl2<e[1] else '≥ '+str(e[1])+' Не'} удовлетворяет
        ''')
        if (usl1 < e[1] and usl2 < e[1]):
            if(float(norm(dictGradDown["x1"][k]-dictGradDown["x1"][k-1],dictGradDown["x2"][k]-dictGradDown["x2"][k-1])) ** 0.5<e[1] and abs(fxk - dictGradDown["f"][k-1])<e[1]):break
        k+=1
    print(f'Ответ k = {len(dictGradDown["x1"])-1}; x {(dictGradDown["x1"][-1],dictGradDown["x2"][-1])} {"в связи с ограничением M" if k>=M else ""} f = {addtoDict(dictGradDown, "f",func(dictGradDown["x1"][-1], dictGradDown["x1"][-1]))}')
    return dictGradDown


def fastgradientDown():
    dictFastGrad,k = {'x1':[x0[0]],'x2':[x0[1]], 'f':[]},0
    while k < M:
        print(f'''
        Метод наискорейшего градиентного спуска итерация (k = {k})
        точка ({dictFastGrad['x1'][k]},{dictFastGrad['x2'][k]})
        градиент функции в точке {(dfxk := derivF(dictFastGrad['x1'][k], dictFastGrad['x2'][k]))}
        Проверка услови оконччания {(gr := float(norm(dfxk[0], dfxk[1])) ** 0.5)} {'< ' + str(e[0]) if gr < e[0] else '≥ ' + str(e[0]) + ' Не'} удовлетворяет
        ''')
        if gr < e[1]: break
        print(f'''
        Текущее значение tk {(tk:=extremumMin(dictFastGrad['x1'][k],dictFastGrad['x2'][k],x1,x1x2,x2))}
        {dictFastGrad['x1'][k]} - {tk[0] } * {float(dfxk[0])}
        x(k+1) = {addtoDict(dictFastGrad,'x1',dictFastGrad['x1'][k] - tk[0] * float(dfxk[0]))}, {addtoDict(dictFastGrad,'x2',dictFastGrad['x2'][k] - tk[1] * float(dfxk[1]))}
        Условие 1: {(usl1 := float(norm(dictFastGrad["x1"][k + 1] - dictFastGrad["x1"][k], dictFastGrad["x2"][k + 1] - dictFastGrad["x2"][k])) ** 0.5)} {'< ' + str(e[1]) if usl1 < e[1] else '≥ ' + str(e[1]) + ' Не'} удовлетворяет
        Условие 2: {(usl2 := abs(
            func(dictFastGrad["x1"][k + 1],dictFastGrad["x2"][k + 1]) - addtoDict(dictFastGrad,'f',func(dictFastGrad["x1"][k],dictFastGrad["x2"][k]))))} {'< ' + str(e[1]) if usl2 < e[1] else '≥ ' + str(e[1]) + ' Не'} удовлетворяет
        ''')
        if (usl1 < e[1] and usl2 < e[1]):
            if (float(norm(dictFastGrad["x1"][k] - dictFastGrad["x1"][k - 1],dictFastGrad["x2"][k] - dictFastGrad["x2"][k - 1])) ** 0.5 < e[1] and abs(dictFastGrad["f"][k] - dictFastGrad["f"][k - 1]) < e[1]): break
        k += 1
    print(
        f'Ответ k = {len(dictFastGrad["x1"]) - 1}; x {(dictFastGrad["x1"][-1], dictFastGrad["x1"][-1])} {"в связи с ограничением M" if k >= M else ""} f = {addtoDict(dictFastGrad, "f",func(dictFastGrad["x1"][-1], dictFastGrad["x1"][-1]))}')
    return dictFastGrad

x1, x1x2, x2 = map(lambda x: getNumber02(f'коэффицент для {x} = ', 'float'), ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])
x0 = list(map(lambda x: getNumber02(f'начальную точку {x} ', 'float'), ["x", "y"]))
e = list(map(lambda x: getNumber02(f'точность {x} ', 'float'), ["e1", "e2"]))
M = getNumber02('предел числа операций', 'int')
#x1,x1x2,x2,x0,e,M=1,0.5,5,(0,0.5),(0.15,0.2),10 #вариант 19
#x1,x1x2,x2,x0,e,M=2,1,1,(0.5,1),(0.1,0.15),10 #пример
z = lambda x, y: "+" + str(x) + y if x > 0 else (str(x) + y if x < 0 else '')
func = lambda x,y: x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2
derivF = lambda x,y: [2*x1 * x + x1x2*y, x1x2*x + y * x2*2]
norm = lambda x,y:x**2+y**2
print(f'Ваша функция f(x) = {("".join(list(map(z, [x1, x1x2, x2], ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])))).lstrip("+").lstrip("1")}')
graphic(gradientDown(),'Метод градиентного спуска с постоянным шагом')
graphic(fastgradientDown(),'Метод быстрого градиентного спуска')
