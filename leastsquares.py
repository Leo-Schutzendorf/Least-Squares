import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify


from scipy.integrate import quad
import scipy
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, \
    parse_expr

x = symbols('x')
inputTarget = input('Enter the target function: ')

'''
inputg1 = input('Enter the first function: ')
inputg2 = input('Enter the second function: ')
inputg3 = input('Enter the third function: ')
'''
# Target function
def targetFunction(text, x):
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    target_function = parse_expr(text, transformations=transformations)
    f = lambdify(symbols('x'), target_function, 'numpy')
    return f(x)

# Example: basis functions
def g1(x): return np.exp(x)
def g2(x): return np.sin(x)
def g3(x): return scipy.special.gamma(x)


basis = [g1, g2, g3]


# Interval
intervala, intervalb = 1, 2


# Compute Gram matrix G_ij = <g_i, g_j>
n = len(basis)
G = np.zeros((n, n))
for i in range(n):
   for j in range(n):
       G[i][j], _ = quad(lambda x: basis[i](x) * basis[j](x), intervala, intervalb)


# Right hand side
rightSide = np.zeros(n)
for var in range(n):
    rightSide[var], _ = quad(lambda x: targetFunction(inputTarget, x) * basis[var](x), intervala, intervalb)


coefficient = np.linalg.solve(G, rightSide)


# Compute approximation error ||f - sum(c_i g_i)||_L2
def approx(x):
   return sum(coefficient[i] * basis[i](x) for i in range(n))


error = np.sqrt(quad(lambda x: (targetFunction(inputTarget, x) - approx(x))**2, intervala, intervalb)[0])


print("Coefficients:", coefficient)
print("L2 approximation error:", error)


xs = np.linspace(intervala, intervalb, 256)
ys_f = [targetFunction(inputTarget, x) for x in xs]
ys_approx = [approx(x) for x in xs]


plt.figure(figsize=(8,5))
plt.plot(xs, ys_f, label="Target function $f(x)$", color="blue")
plt.plot(xs, ys_approx, label="Approximation", color="red", linestyle="--")
plt.title("Best L2 Approximation of f(x) on [1,2]")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
