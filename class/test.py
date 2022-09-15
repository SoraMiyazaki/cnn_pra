import numpy as np
import matplotlib.pyplot as plt
import time

# 計算条件
rho = 1000 #[kg/m^3]
nu = 10 ** (-6) #[m^2/s]
g = 9.8 #[m/s^2]
re = 400
L = 0.02 #[m]
Us = 0.02 #[m/s]
mx = 68
my = 68
dx = L / (mx - 4)
dy = L / (my - 4)
dt = 10 **(-3)
eps = 10 ** (-6)
ktend = 1000
kend = 100000
psum = 0
x = np.linspace(0, 2, 64)
y = np.linspace(0, 2, 64)
X, Y = np.meshgrid(x, y)
p_min = 0
p_max = 1.0
interval = np.linspace(p_min, p_max, 11)

# 初期条件
u = np.zeros((my, mx))
v = np.zeros((my + 1, mx))
p = np.zeros((my, mx))
p[1:-2, 2:-2] = 0.4
v4 = np.zeros((my, mx))
u4 = np.zeros((my, mx))

# uの定義点であるv4
def func_v4():
    v4[2:-2, 3:-2] = (1/4) * (v[3:-2, 3:-2] + v[3:-2, 2:-3] + v[2:-3, 3:-2] + v[2:-3, 2:-3])
    return v4

# vの定義点であるu4
def func_u4():
    u4[1:-3, 2:-2] = (1/4) * (u[1:-3, 2:-2] + u[1:-3, 3:-1] + u[2:-2, 2:-2] + u[2:-2, 3:-1])
    return u4      

# NSeqの計算(新しいu)
# 移流項の計算
def func_uat1():
    uat1 = (((u[2:-2, 3:-2] + np.abs(u[2:-2, 3:-2])) / 2) * ((u[2:-2, 3:-2]) - (u[2:-2, 2:-3]) / dx)) \
        + (((u[2:-2, 3:-2] - np.abs(u[2:-2, 3:-2])) / 2) * ((u[2:-2, 4:-1] - u[2:-2, 3:-2]) / dx)) 
    return uat1      

def func_uat2():
    uat2 = (((func_v4()[2:-2, 3:-2] + np.abs(func_v4()[2:-2, 3:-2] )) / 2) * ((u[2:-2, 3:-2]) - (u[3:-1, 3:-2]) / dy)) \
        + (((func_v4()[2:-2, 3:-2] - np.abs(func_v4()[2:-2, 3:-2] )) / 2) * ((u[1:-3, 3:-2] - u[2:-2, 3:-2]) / dy))  
    return uat2

# 圧力項の計算
def func_upt():
    upt = (1/rho) * ((p[2:-2, 3:-2] - p[2:-2, 2:-3]) / dx)
    return upt  
 
# 拡散項の計算
def func_udt():
    udt = nu * (((u[2:-2, 4:-1] - (2 * u[2:-2, 3:-2]) + u[2:-2, 2:-3]) / dx ** 2) \
        + ((u[1:-3, 3:-2] - (2 * u[2:-2, 3:-2]) + u[3:-1, 3:-2]) / dy ** 2))
    return udt    

def func_unew():
    unew = u[2:-2, 3:-2] - (dt * (func_uat1() + func_uat2() + func_upt() - func_udt()))
    return unew

# NSeqの計算(新しいv)
# 移流項の計算
def func_vat1():
    vat1 = (((func_u4()[1:-3, 2:-2] + np.abs(func_u4()[1:-3, 2:-2] )) / 2) * ((v[2:-3, 2:-2]) - (v[2:-3, 1:-3]) / dx)) \
        + (((func_u4()[1:-3, 2:-2] - np.abs(func_u4()[1:-3, 2:-2] )) / 2) * ((v[2:-3, 3:-1] - v[2:-3, 2:-2]) / dx))  
    return vat1

def func_vat2():
    vat2 = (((v[2:-3, 2:-2] + np.abs(v[2:-3, 2:-2])) / 2) * ((v[2:-3, 2:-2]) - (v[3:-2, 2:-2]) / dy)) \
        + (((v[2:-3, 2:-2] - np.abs(v[2:-3, 2:-2])) / 2) * ((v[1:-4, 2:-2] - v[2:-3, 2:-2]) / dy)) 
    return vat2  

# 圧力項の計算
def func_vpt():
    vpt = (1/rho) * ((p[1:-3, 2:-2] - p[2:-2, 2:-2]) / dx)
    return vpt  

# 拡散項の計算
def func_vdt():
    vdt = nu * (((v[2:-3, 3:-1] - (2 * v[2:-3, 2:-2]) + v[2:-3, 1:-3]) / dx ** 2) \
        + ((v[1:-4, 2:-2] - (2 * v[2:-3, 2:-2]) + v[3:-2, 2:-2]) / dy ** 2))
    return vdt    

def func_vnew():
    vnew = v[2:-3, 2:-2] - dt * (func_vat1() + func_vat2() + func_vpt() - func_vdt())
    return vnew   

# 圧力の計算
def func_pp1():
    pp1 = (((u[1:-2, 3:-1] - u[1:-2, 2:-2]) / dx) + ((v[1:-3, 2:-2] - v[2:-2, 2:-2]) / dy)) / dt
    return pp1

def func_pp2():
    pp2 = -(((u[1:-2, 3:-1] * ((u[1:-2, 4:] - u[1:-2, 2:-2]) / (2 * dx))) \
        + (func_v4()[1:-2, 3:-1] * ((u[-3, 3:-1] - u[2:-1, 3:-1]) / (2 * dy)))) \
            - ((u[1:-2, 2:-2] * ((u[1:-2, 3:-1] - u[1:-2, 1:-3]) / (2 * dx))) \
                + (func_v4()[1:-2, 2:-2] * ((u[:-3, 2:-2] - u[2:-1, 2:-2]) / (2 * dy))))) / dx
    return pp2

def func_pp3():
    pp3 = -(((func_u4()[:-3, 2:-2] * ((v[1:-3, 3:-1] - v[1:-3, 1:-3]) / (2 * dx))) \
        + (v[1:-3, 2:-2] * ((v[:-4, 2:-2] - v[2:-2, 2:-2]) / (2 * dy)))) \
            - ((func_u4()[1:-2, 2:-2] * ((v[2:-2, 3:-1] - v[2:-2, 1:-3]) / (2 * dx))) \
                + (v[2:-2, 2:-2] * ((v[1:-3, 2:-2] - v[3:-1, 2:-2]) / (2 * dy))))) / dy
    return pp3

def func_phi():
    phi = rho * (func_pp1() + func_pp2() + func_pp3())    
    return phi  

def func_pnew():
    pnew = (((p[1:-2, 3:-1] + p[1:-2, 1:-3] + p[:-3, 2:-2] + p[2:-1, 2:-2]) / 4) - (((dx ** 2) / 4) * func_phi()))
    return pnew

def func_BorderTerms_u():
    # 左
    u[2:-2, 2] = 0
    u[2:-2, 1] = u[2:-2, 2]
    # 右
    u[2:-2, -2] = 0
    u[2:-2, -1] = u[2:-2, -2]
    # 上
    u[1, :] = Us
    u[0, :] = u[1, :]
    # 下
    u[-2, :] = -u[-3, :]
    return u    

def func_BorderTerms_v():
    # 上
    v[2, 2:-2] = 0
    v[1, 2:-2] = v[2, 2:-2]
    v[0, 2:-2] = v[1, 2:-2]
    # 下
    v[-3, 2:-2] = 0
    v[-2, 2:-2] = v[-3, 2:-2]
    # 左
    v[:, 1] = -v[:, 2]
    # 右
    v[:, -2] = -v[:, -3]
    return v

def func_BorderTerms_p():
    # 上
    p[1, 2:-2] = p[2, 2:-2]
    p[0, 2:-2] = p[1, 2:-2]
    # 下
    p[-2, 2:-2] = p[-3, 2:-2]
    # 左
    p[2:-2, 1] = p[2:-2, 2]
    # 右
    p[2:-2, -2] = p[2:-2, -3]
    p[2:-2, -1] = p[2:-2, -2] 
    return p
    

# 時間発展の計算(メイン)
count = 0
for i in range(2):
    count += 1
    if count == 1:
        print(count, "st")
    elif count == 2:
        print(count, "nd")
    elif count == 3:
        print(count, "rd")
    else:
        print(count, "th")

    # 境界条件
    func_BorderTerms_u()
    func_BorderTerms_v()
    func_BorderTerms_p()

    # NSeq
    func_v4()
    func_u4()
    func_uat1()
    func_uat2()
    func_upt()
    func_udt()
    func_vat1()
    func_vat2()
    func_vpt()
    func_vdt()
    unew = func_unew()
    vnew = func_vnew()

    np.savetxt("u.csv", unew)
    np.savetxt("v.csv", vnew)

    # unewをuに上書き
    u[2:-2, 3:-2] = unew

    # vnewをvに上書き
    v[2:-3, 2:-2] = vnew

    # 境界条件
    func_BorderTerms_u()
    func_BorderTerms_v()
    func_BorderTerms_p()

    # v4, u4の更新
    func_u4()
    func_v4()

    u4[1, 2:-2] = u4[2, 2:-2]
    v4[1:-2, -2] = -v4[1:-2, -3]

    # Peq
    knum = 0
    time_sta = time.time()
    while True:

        func_pp1()
        func_pp2()
        func_pp3()
        func_phi()
        func_pnew()

        # 圧力の二乗誤差
        psum = np.sum((func_pnew() - p[1:-2, 2:-2]) ** 2)

        # 二乗誤差の和がepsより大きかったら以下を繰り返す
        if psum > eps:
            p[1:-2, 2:-2] = func_pnew()
        else:
            break

        knum += 1

    np.savetxt("p.csv", func_pnew())   
    time_end = time.time()

    # 約0.013s
    Time = (time_end - time_sta)
    print(psum, knum,"回",  Time, "s")  

    fig = plt.figure(figsize=(11,7))
    plt.contourf(X, Y, p[2:-2, 2:-2], interval, extend="both", alpha=0.5)  
    plt.colorbar()
    plt.streamplot(X, Y, u[2:-2, 2:-2], v[3:-2, 2:-2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("figure.png")


