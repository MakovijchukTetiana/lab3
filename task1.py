import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return np.sin(x + 1.5) - y - 1.0

def f2(x, y):
    return np.cos(y - 1.5) + 2.0 * x

def jacobian(x, y):
    df1_dx = np.cos(x + 1.5)
    df1_dy = -1.0
    df2_dx = 2.0
    df2_dy = -np.sin(y - 1.5)

    return np.array([
        [df1_dx, df1_dy],
        [df2_dx, df2_dy]
    ])

def newton_method_system(x0, y0, tol=1e-3, max_iter=50):
    x, y = x0, y0

    print(
        f"{'k':<3} | {'x(k)':<10} | {'y(k)':<10} | {'|x(k)-x(k-1)|':<15} | {'|y(k)-y(k-1)|':<15} | {'max_delta':<10} | {'Висновок'}")
    print("-" * 88)
    print(f"{0:<3} | {x:<10.5f} | {y:<10.5f} | {'-':<15} | {'-':<15} | {'-':<10} | -")

    for k in range(1, max_iter + 1):
        F = np.array([-f1(x, y), -f2(x, y)])
        W = jacobian(x, y)

        delta = np.linalg.solve(W, F)

        x_new = x + delta[0]
        y_new = y + delta[1]

        delta_x = abs(x_new - x)
        delta_y = abs(y_new - y)
        max_delta = max(delta_x, delta_y)

        status = f"<{tol}" if max_delta <= tol else f">{tol}"
        print(
            f"{k:<3} | {x_new:<10.5f} | {y_new:<10.5f} | {delta_x:<15.5f} | {delta_y:<15.5f} | {max_delta:<10.5f} | {status}")

        if max_delta <= tol:
            print("-" * 88)
            print(f"Розв'язок знайдено за {k} ітерацій!")
            print(f"x* = {x_new:.5f} ≈ {round(x_new, 3)}")
            print(f"y* = {y_new:.5f} ≈ {round(y_new, 3)}")

            print("\nПеревірка (підстановка коренів у систему):")
            print(f"f1(x*, y*) = {f1(x_new, y_new):.6f}")
            print(f"f2(x*, y*) = {f2(x_new, y_new):.6f}")
            return x_new, y_new

        x, y = x_new, y_new

    print("Метод не зійшовся за задану кількість ітерацій.")
    return None

def plot_system(root=None):
    
    x_vals = np.linspace(-3.0, 3.0, 500)
    y_vals = np.linspace(-3.0, 3.0, 500)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    plt.figure(figsize=(9, 7))

    plt.contour(X, Y, Z1, levels=[0], colors='red', linewidths=2)
    plt.contour(X, Y, Z2, levels=[0], colors='green', linewidths=2)

    plt.plot([], [], color='red', linewidth=2, label=r'$f_1(x, y) = \sin(x + 1.5) - y - 1 = 0$')
    plt.plot([], [], color='green', linewidth=2, label=r'$f_2(x, y) = \cos(y - 1.5) + 2x = 0$')

    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    if root is not None:
        rx, ry = root
        plt.plot(rx, ry, 'bo', markersize=7, label=f'Корінь: ({rx:.3f}, {ry:.3f})')

        d = 0.35
        rect_x = [rx - d, rx + d, rx + d, rx - d, rx - d]
        rect_y = [ry - d, ry - d, ry + d, ry + d, ry - d]
        plt.plot(rect_x, rect_y, 'b--', linewidth=1.2, label='Область локалізації G')
        plt.text(rx + d + 0.05, ry + d, '$G$', fontsize=12, color='blue', weight='bold')

    plt.title("Рис. 1. Локалізація розв'язку системи", fontsize=13, pad=20)
    plt.xlabel('x', loc='right')
    plt.ylabel('y', loc='top', rotation=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

epsilon = 1e-3
initial_x = 0.0
initial_y = 0.0

print("Метод Ньютона для розв'язування системи нелінійних рівнянь")
root = newton_method_system(initial_x, initial_y, tol=epsilon)

plot_system(root)
