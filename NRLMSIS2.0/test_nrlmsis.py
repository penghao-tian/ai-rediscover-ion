import ctypes
import numpy as np
# import numpy.ctypeslib as npct

# 加载共享库
libmsis = ctypes.CDLL('./NRLMSIS2.0/libmsis.cpython-39-x86_64-linux-gnu.so')


# gtd8d = libmsis.gtd8d

# gtd8d.argtypes= [npct.ndpointer(dtype=np.int32, ndim=0, flags="F"),   # iyd
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # sec
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # alt 
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # glat
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # glong
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # stl
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # f107a
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # f107
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # ap array
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # mass 
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # d (array)
#                  npct.ndpointer(dtype=np.float32, ndim=0, flags="F"), # t (array)
#                  ]
# 定义子程序接口
libmsis.gtd8d.argtypes = [
    ctypes.c_int,            # iyd
    ctypes.c_float,          # sec
    ctypes.c_float,          # alt
    ctypes.c_float,          # glat
    ctypes.c_float,          # glong
    ctypes.c_float,          # stl
    ctypes.c_float,          # f107a
    ctypes.c_float,          # f107
    ctypes.POINTER(ctypes.c_float),  # ap (array)
    ctypes.c_int,            # mass
    ctypes.POINTER(ctypes.c_float),  # d (array)
    ctypes.POINTER(ctypes.c_float)   # t (array)
]
libmsis.gtd8d.restype = None

# 定义输入参数
iyd = ctypes.c_int(20210)
sec = ctypes.c_float(36000.0)
alt = ctypes.c_float(150.0)
glat = ctypes.c_float(45.0)
glong = ctypes.c_float(-93.0)
stl = ctypes.c_float(10.0)
f107a = ctypes.c_float(150.0)
f107 = ctypes.c_float(150.0)
ap = (ctypes.c_float * 7)(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
mass = ctypes.c_int(1)

# 定义输出参数
d = (ctypes.c_float * 9)()
t = (ctypes.c_float * 2)()

# print(iyd)
# print(ap)
# print(d)
# breakpoint()

# 调用Fortran子程序
libmsis.gtd8d(iyd, sec, alt, glat, glong, stl, f107a, f107, ap, mass, d, t)

# # 打印输出结果
print("Density (d):", [d[i] for i in range(9)])
print("Temperature (t):", [t[i] for i in range(2)])


# gfortran -O3 -cpp -shared -fPIC -o libmsis2.0.libso msis_gtd8d_iso_c_binding.F90 alt2gph.F90 msis_calc.F90 msis_constants.F90 msis_dfn.F90 msis_gfn.F90 msis_tfn.F90 msis_init.F90

# gfortran -O3 -cpp -shared -fPIC -o libmsis2.0.libso alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90 msis_gtd8d_iso_c_binding.F90

# fortran -O3 -cpp -o msis2.0_test.exe alt2gph.F90 msis_constants.F90
      # msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90
      # msis_gtd8d.F90 msis2.0_test.F90

# f2py -c -m libmsis msis_gtd8d_iso_c_binding.F90 alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90
# f2py -c -m libmsis msis_gtd8d.F90 alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90



# gfortran -O3 -cpp -o msis2.0_test.exe alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90 msis_gtd8d.F90 msis2.0_test.F90