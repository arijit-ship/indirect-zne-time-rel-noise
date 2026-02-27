import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def model_expression_stage1(C, F1):
    """
    Mathematical Expression for Stage 1:
    R(C) = F1 * C
    """
    return F1 * C

def model_expression_stage2(T, F10, F11, F12):
    """
    Mathematical Expression for Stage 2:
    F1(T) = F10 + F11*T + F12*T^2
    """
    return F10 + F11*T + F12*T**2

def fit_stage1(C, R, degree=1, include_intercept=False):
    """
    Fits R vs C to find the coefficient F1.
    By default, degree=1 and include_intercept=False (R = F1*C).
    """
    C_reshaped = np.array(C).reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=degree, include_bias=include_intercept)
    X_poly = poly.fit_transform(C_reshaped)
    
    model = LinearRegression(fit_intercept=include_intercept)
    model.fit(X_poly, R)
    
    # Returns the coefficients (F1, etc.)
    return model.coef_

def fit_stage2(T, F1_values, degree=2, include_intercept=True):
    """
    Fits F1 vs T to find global constants F10, F11, F12.
    By default, degree=2 and include_intercept=True (F1 = F10 + F11*T + F12*T^2).
    """
    T_reshaped = np.array(T).reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=degree, include_bias=include_intercept)
    X_poly = poly.fit_transform(T_reshaped)
    
    # We set fit_intercept=False because PolynomialFeatures(include_bias=True) 
    # already creates the intercept column for us.
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, F1_values)
    
    return model.coef_