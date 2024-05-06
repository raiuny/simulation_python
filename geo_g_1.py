from sympy import symbols, diff


if __name__ == "__main__":
    x = symbols("x")
    y = (x-1)* x **2 / (x-1+1/3*(1-x**2))
    dy_dx = diff(y,x)
    print(dy_dx.subs(x,1))