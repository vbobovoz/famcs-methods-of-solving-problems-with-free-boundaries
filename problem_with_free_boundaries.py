import matplotlib.pyplot as plt
import numpy as np

I_value = 0.0

def solve(zPrev, P, alpha):
    r = np.linspace(0, 1, N + 1)
    h = 1.0 / N
    I = 2 * np.pi * h * np.sum(r[1:N] * zPrev[1:N])
    C = (P / (2 * I)) - 2 * np.sin(np.radians(alpha))

    a_buffer = np.zeros(N + 1)
    a_buffer[1:N] = (r[1:N] - (h / 2)) / np.sqrt(1 + ((zPrev[1:N] - zPrev[:N - 1]) / h) ** 2)

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    f = np.zeros(N + 1)

    a[0] = 0
    c[0] = 1 / h
    b[0] = 1 / h
    f[0] = C * h / 2

    a[N - 1] = 0
    c[N - 1] = 1
    b[N - 1] = 0
    f[N - 1] = (h * np.tan(np.radians(alpha)) +
                (h ** 2 / 2) * (1 / (np.cos(np.radians(alpha)) ** 3)) *
                (-(P / I) + C + np.sin(np.radians(alpha))))

    a[N] = 0
    c[N] = 1
    f[N] = 0

    for i in range(1, N - 1):
        a[i] = a_buffer[i] / (h ** 2)
        c[i] = (a_buffer[i + 1] + a_buffer[i]) / (h ** 2)
        b[i] = a_buffer[i + 1] / (h ** 2)
        f[i] = ((P * (r[i] ** 3)) / I) - (C * r[i])

    global I_value
    I_value = I

    return sweepMethod(N, a, b, c, f)

def sweepMethod(N, a, b, c, f):
    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 2)

    alpha[1] = b[0] / c[0]
    for i in range(1, N):
        alpha[i + 1] = b[i] / (c[i] - alpha[i] * a[i])

    beta[1] = f[0] / c[0]
    for i in range(1, N + 1):
        beta[i + 1] = (f[i] + a[i] * beta[i]) / (c[i] - a[i] * alpha[i])

    y = np.zeros(N + 1)
    y[N] = beta[N + 1]
    for i in range(N - 1, -1, -1):
        y[i] = alpha[i + 1] * y[i + 1] + beta[i + 1]

    return y

if __name__ == "__main__":
    N = 100 # Число разбиений 
    eps = 1e-6 # Точность
    rho = 1.0 # г/см^3
    sigma = 72.75 # дин/см
    omega = 0.75 # 1/сек
    V = 1.02 # см^3
    alpha = 60 # градусов

    # ----------------------------------------------------------------------
    P = omega**2 * rho * V / (2 * sigma)
    print(f"P0 = {P}")

    zNext = np.linspace(0, 1, N + 1)
    zNext = 1 - zNext
    zPrev = np.zeros_like(zNext)

    while np.max(np.abs(zNext - zPrev)) > eps:
        zPrev = zNext.copy()
        zNext = solve(zPrev, P, alpha)

    print(f"I0 = {I_value}")
    
    r0 = [i * 1/N / (I_value ** (1/3)) for i in range(N + 1)]
    z0 = [zNext[i] / (I_value ** (1/3)) for i in range(N + 1)]
    # ----------------------------------------------------------------------
    print()
    # ----------------------------------------------------------------------
    omega = 0.75*10
    P = omega**2 * rho * V / (2 * sigma)
    print(f"P1 = {P}")

    zNext = np.linspace(0, 1, N + 1)
    zPrev = np.zeros_like(zNext)

    while np.max(np.abs(zNext - zPrev)) > eps:
        zPrev = zNext.copy()
        zNext = solve(zPrev, P, alpha)

    print(f"I1 = {I_value}")
    
    r1 = [i * 1/N / (I_value ** (1/3)) for i in range(N + 1)]
    z1 = [zNext[i] / (I_value ** (1/3)) for i in range(N + 1)]
    # ----------------------------------------------------------------------
    print()
    # ----------------------------------------------------------------------
    omega = 0.75*15
    P = omega**2 * rho * V / (2 * sigma)
    print(f"P2 = {P}")

    zNext = np.linspace(0, 1, N + 1)
    zPrev = np.zeros_like(zNext)

    while np.max(np.abs(zNext - zPrev)) > eps:
        zPrev = zNext.copy()
        zNext = solve(zPrev, P, alpha)

    print(f"I2 = {I_value}")
    
    r2 = [i * 1/N / (I_value ** (1/3)) for i in range(N + 1)]
    z2 = [zNext[i] / (I_value ** (1/3)) for i in range(N + 1)]
    # ----------------------------------------------------------------------
    print()
    # ----------------------------------------------------------------------
    omega = 0.75*20
    P = omega**2 * rho * V / (2 * sigma)
    print(f"P3 = {P}")

    zNext = np.linspace(0, 1, N + 1)
    zPrev = np.zeros_like(zNext)

    while np.max(np.abs(zNext - zPrev)) > eps:
        zPrev = zNext.copy()
        zNext = solve(zPrev, P, alpha)

    print(f"I3 = {I_value}")

    r3 = [i * 1/N / (I_value ** (1/3)) for i in range(N + 1)]
    z3 = [zNext[i] / (I_value ** (1/3)) for i in range(N + 1)]
    # ----------------------------------------------------------------------

    plt.plot(r0, z0, linestyle='-', label='$P_0 = 0.003942989$')
    plt.plot(r1, z1, linestyle='--', label='$P_1 = 0.394329896$')
    plt.plot(r2, z2, linestyle='-.', label='$P_2 = 0.887242268$')
    plt.plot(r3, z3, linestyle=':', label='$P_3 = 1.577319587$')
    plt.title('График поведения капли жидкости при разных значениях параметра $P$')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.grid(True)
    plt.legend()
    plt.show()