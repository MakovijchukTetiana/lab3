import math

def phi1(x, y):
    return -0.5 * math.cos(y - 1.5)

def phi2(x, y):
    return math.sin(x + 1.5) - 1.0


def seidel_method(x0, y0, tol=1e-3, max_iter=50):
    print(" " * 32 + "МЕТОД ЗЕЙДЕЛЯ")
    print(
        f"{'k':<3} | {'x(k)':<10} | {'y(k)':<10} | {'|x(k)-x(k-1)|':<15} | {'|y(k)-y(k-1)|':<15} | {'max_delta':<10} | {'Висновок'}")
    print("-" * 88)

    x, y = x0, y0
    print(f"{0:<3} | {x:<10.5f} | {y:<10.5f} | {'-':<15} | {'-':<15} | {'-':<10} | -")

    for k in range(1, max_iter + 1):
        x_new = phi1(x, y)

        y_new = phi2(x_new, y)

        delta_x = abs(x_new - x)
        delta_y = abs(y_new - y)
        max_delta = max(delta_x, delta_y)

        status = f"<{tol}" if max_delta <= tol else f">{tol}"
        print(
            f"{k:<3} | {x_new:<10.5f} | {y_new:<10.5f} | {delta_x:<15.5f} | {delta_y:<15.5f} | {max_delta:<10.5f} | {status}")

        if max_delta <= tol:
            print("-" * 88)
            print(f"Метод Зейделя зійшовся за {k} ітерацій!")
            print(f"x* = {x_new:.5f} ≈ {round(x_new, 3)}")
            print(f"y* = {y_new:.5f} ≈ {round(y_new, 3)}")
            return x_new, y_new

        x, y = x_new, y_new

    print("Метод Зейделя не зійшовся за задану кількість ітерацій.")
    return None

epsilon = 1e-3
initial_x = 0.0
initial_y = 0.0

seidel_method(initial_x, initial_y, tol=epsilon)
