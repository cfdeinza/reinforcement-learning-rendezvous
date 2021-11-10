from scipy.integrate import solve_ivp
from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, exp


def func(t, y):

    # dy = cos(t)
    dy = y

    return dy


t_span = [0, 5]
# t_span = np.linspace(0, 5, 501)
y0 = np.array([1])
rtol, atol = (1e-8, 1e-8)

# sol = solve_ivp(func, t_span, y0)
sol = solve_ivp(func, t_span, y0, rtol=rtol, atol=atol)

t = sol.t
y = sol.y

print('ivp:')
print(t)
print(y)
print(len(t))

# rk = RK45(func, 0., y0, 5.)
rk = RK45(func, 0., y0, 5., rtol=rtol, atol=atol)

rkt = [rk.t]
rky = [rk.y]

for i in range(100):
    rk.step()
    rkt.append(rk.t)
    rky.append(rk.y)
    if rk.status == 'finished':
        break

print('RK45:')
print(rkt)
print(rky)
print(len(rkt))

fig = plt.figure(num=1, clear=True)

plt.plot(t, y.reshape(t.shape), 'b', label='ivp')
plt.plot(rkt, rky, 'r', label='RK45')
# plt.plot(rkt, [sin(i) for i in rkt], 'k', label='sin')
plt.plot(rkt, [exp(i) for i in rkt], 'k--', label='exp')
plt.grid()
plt.legend()
plt.show()

print('Finished.')
