##perceptron


def AND(x, y):
    tmp = x + y - 1
    if tmp > 0:
        return 1
    else:
        return 0


def NAND(x, y):
    tmp = -x - y + 2
    if tmp > 0:
        return 1
    else:
        return 0


def OR(x, y):
    tmp = x + y
    if tmp > 0:
        return 1
    else:
        return 0


print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))
print(NAND(0, 0), NAND(0, 1), NAND(1, 0), NAND(1, 1))
print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))
