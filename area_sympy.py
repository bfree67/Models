# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:21:44 2021

@author: Brian
"""
from sympy import *
from sympy.functions import exp, log, erfc
from sympy.abc import x, y, d
from sympy import pi

ay, by, cy, sig_y = symbols('ay by cy sig_y')
az, bz, cz, sig_z = symbols('az bz cz sig_z')

y_expr, x_expr, y_integrand, x_integrand = symbols('y_expr x_expr y_integrand x_integrand')

sig_y = ay*(x**(by+cy*log(x))) ## coefficient for the cross-wind distribution (y-axis) output in [m]
sig_z = az*(x**(bz+cz*log(x))) ## coefficient for the vertical distribution (z-axis) output in [m]

y_expr = erfc(y/sig_y)

x_expr = (d/(sig_y * sig_z)) * y_expr
          
x_integrand = integrate(x_expr, x)