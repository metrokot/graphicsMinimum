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

def newtonMethod():
    pass


#x1, x1x2, x2 = map(lambda x: getNumber02(f'коэффицент для {x} = ', 'int'), ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])
#x0 = list(map(lambda x: getNumber02(f'начальную точку {x} ', 'float'), ["x", "y"]))
#e = list(map(lambda x: getNumber02(f'точность {x} ', 'float'), ["e1", "e2"]))
#M = getNumber02('предел числа операций', 'int')
x1,x1x2,x2,x0,e,M=1,0.5,5,(0,0.5),(0.15,0.2),10
#x1,x1x2,x2,x0,e,M=2,1,1,(0.5,1),(0.1,0.15),10
z = lambda x, y: "+" + str(x) + y if x > 0 else (str(x) + y if x < 0 else '')
func = lambda x,y: x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2
derivF = lambda x,y: [2*x1 * x + x1x2*y, x1x2*x + y * x2*2]
norm = lambda x,y:x**2+y**2
print(f'Ваша функция f(x) = {("".join(list(map(z, [x1, x1x2, x2], ["(x_1)^2", "(x_1)*(x_2)", "(x_2)^2"])))).lstrip("+").lstrip("1")}')
