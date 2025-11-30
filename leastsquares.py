import numpy as np
import matplotlib.pyplot as plt
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
x_sym = symbols('x')

print("Enter expressions using standard Python/math notation.")
print("Examples: sin(x), exp(-x), x**2, log(x+17), etc.\n")

inputTarget = input('Enter the target function f(x): ').strip()

inputg1 = input('Enter the first basis function g1(x): ').strip()
inputg2 = input('Enter the second basis function g2(x): ').strip()
inputg3 = input('Enter the third basis function g3(x): ').strip()


# ----------------------------
# Step 2: Helper to turn a string into a fast numpy callable
# ----------------------------
def string_to_callable(expr_str):
    """
    Convert a string like "sin(x) + x**2" into a function(x) -> float/array.
    """
    transformations = (standard_transformations +
                       (implicit_multiplication_application, convert_xor))
    expr = parse_expr(expr_str, transformations=transformations)
    func = lambdify(x_sym, expr, modules=['numpy', 'scipy'], dummify=False)
    return func


# Create the actual callable functions
f  = string_to_callable(inputTarget)   # target function
g1 = string_to_callable(inputg1)       # <-- now really user-defined
g2 = string_to_callable(inputg2)       # <-- now really user-defined
g3 = string_to_callable(inputg3)       # <-- now really user-defined

basis = [g1, g2, g3]   # list of the three user-defined basis functions


# ----------------------------
# Step 3: Define interval (you can change it if you want)
# ----------------------------
a, b = 0, 16         # choose an interval where all functions are well-behaved
print(f"\nUsing approximation interval [{a}, {b}]")


# ----------------------------
# Step 4: Compute Gram matrix G_ij = ∫ g_i(x) g_j(x) dx
# ----------------------------
n = len(basis)
G = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):                 # symmetric matrix → compute only upper part
        integrand = lambda x: basis[i](x) * basis[j](x)
        integral, _ = quad(integrand, a, b, points=[0])  # points=[0] helps with possible singularities at 0
        G[i, j] = G[j, i] = integral


# ----------------------------
# Step 5: Compute right-hand side vector b_j = ∫ f(x) g_j(x) dx
# ----------------------------
b_vec = np.zeros(n)
for j in range(n):
    integrand = lambda x: f(x) * basis[j](x)
    integral, _ = quad(integrand, a, b, points=[0])
    b_vec[j] = integral


# ----------------------------
# Step 6: Solve the normal equations G c = b for the coefficients c
# ----------------------------
try:
    coeffs = np.linalg.solve(G, b_vec)
except np.linalg.LinAlgError:
    print("Gram matrix is singular – basis functions are linearly dependent on the interval.")
    raise

print("\nCoefficients c0, c1, c2 (for g1, g2, g3):", coeffs)


# ----------------------------
# Step 7: Define the approximation and compute L2 error
# ----------------------------
def approximation(x):
    return coeffs[0]*g1(x) + coeffs[1]*g2(x) + coeffs[2]*g3(x)

# L2 error ||f - approx|| on [a,b]
error_integrand = lambda x: (f(x) - approximation(x))**2
l2_error = np.sqrt(quad(error_integrand, a, b)[0])

print(f"L2 approximation error on [{a},{b}]: {l2_error:.10f}")


# ----------------------------
# Step 8: Plot the result
# ----------------------------
xs = np.linspace(a, b, 800)
ys_target = f(xs)
ys_approx = approximation(xs)

plt.figure(figsize=(10, 6))
plt.plot(xs, ys_target, label=r"Target $f(x)$", linewidth=2)
plt.plot(xs, ys_approx, '--', label=r"Approximation $c_1 g_1 + c_2 g_2 + c_3 g_3$", linewidth=2)
plt.title(f"Least-Squares Approximation with User-Defined Basis\n"
          f"Error = {l2_error:.2e}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()