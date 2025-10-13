import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    parse_expr,
)

# ----------------------------
# Step 1: User input
# ----------------------------
x = symbols('x')
inputTarget = input('Enter the target function: ')



# User-specified basis functions
inputg1 = input('Enter the first function: ')
inputg2 = input('Enter the second function: ')
inputg3 = input('Enter the third function: ')


# ----------------------------
# Step 2: Define target function parser
# ----------------------------
def targetFunction(text, x):
    """
    Convert a string expression into a callable numpy function.
    Example input: "sin(x) + exp(x)"
    """
    transformations = (standard_transformations +
                       (implicit_multiplication_application, convert_xor))
    target_expr = parse_expr(text, transformations=transformations)
    f = lambdify(symbols('x'), target_expr, 'numpy')
    return f(x)

# ----------------------------
# Step 3: Define basis functions
# ----------------------------
def g1(x): return np.exp(x)
def g2(x): return np.sin(x)
def g3(x): return scipy.special.gamma(x)

basis = [g1, g2, g3]   # List of basis functions

# ----------------------------
# Step 4: Define interval
# ----------------------------
intervala, intervalb = 1, 2   # Approximation interval

# ----------------------------
# Step 5: Compute Gram matrix
# ----------------------------
# Gram matrix G_ij = <g_i, g_j> over [intervala, intervalb]
n = len(basis)
G = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        G[i][j], _ = quad(lambda x: basis[i](x) * basis[j](x),
                          intervala, intervalb)

# ----------------------------
# Step 6: Compute RHS vector
# ----------------------------
# Right-hand side: b_j = <f, g_j>
rightSide = np.zeros(n)
for j in range(n):
    rightSide[j], _ = quad(lambda x: targetFunction(inputTarget, x) * basis[j](x),
                           intervala, intervalb)

# ----------------------------
# Step 7: Solve linear system
# ----------------------------
# Solve G * c = b for coefficients
coefficient = np.linalg.solve(G, rightSide)

# ----------------------------
# Step 8: Approximation + error
# ----------------------------
def approx(x):
    """Return approximation sum(c_i * g_i(x))"""
    return sum(coefficient[i] * basis[i](x) for i in range(n))

# L2 error over [intervala, intervalb]
error = np.sqrt(quad(lambda x: (targetFunction(inputTarget, x) - approx(x))**2,
                     intervala, intervalb)[0])

print("Coefficients:", coefficient)
print("L2 approximation error:", error)

# ----------------------------
# Step 9: Plot results
# ----------------------------
xs = np.linspace(intervala, intervalb, 256)
ys_f = [targetFunction(inputTarget, x) for x in xs]
ys_approx = [approx(x) for x in xs]

plt.figure(figsize=(8, 5))
plt.plot(xs, ys_f, label="Target function $f(x)$", color="blue")
plt.plot(xs, ys_approx, label="Approximation", color="red", linestyle="--")
plt.title("Best L2 Approximation of f(x) on [1,2]")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
