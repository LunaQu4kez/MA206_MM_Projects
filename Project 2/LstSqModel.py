import numpy as np

'''
    The basic model of a least-square model.
    
    The model use the numpy.lstsq() to solve a least problem, which requires two important parameters X and w, y = Xw
    
    All the problem should transformed into the form, thus the transX, transY methods provide the transformation.
'''
class LstSqModel():
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    
    def __init__(self, x, y) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        
    
    def transX(self):
        raise NameError("Undefined method.")

    def transY(self):
        raise NameError("Undefined method.")


    def fit_transform(self):
        self.transX()
        self.transY()
        self.w = np.linalg.lstsq(self.x, self.y, rcond=None)[0]
        return self.w


'''
    The extend model with y = c_0 + c_1 x + c_2 x^2 + \cdots c_n x^n, which return the parameter w = (c_0, c_1, ..., c_n).
    
    One should provide the degree of a polynomial form.
    
    Then the X will be
        [x.^0, x.^1, ..., x.^{n - 1}]
    where the '.^' representes the vectorilized power function.
'''
class PolyModel(LstSqModel):
    degree: int
    
    def __init__(self, x, y, n: int) -> None:
        super().__init__(x, y)
        self.degree = n
        self.w = np.ndarray(shape=(self.degree + 1, ))
    
    def transX(self):
        _X = np.repeat(self.x.reshape((-1, 1)), repeats=self.degree + 1, axis=1)
        _p = np.arange(self.degree + 1)
        self.x = np.power(_X, _p)
        return self.x
        
    def transY(self):
        self.y = self.y
        return self.y
        

'''
    The model with y = ax^n, which return the parameter w = a.
'''
class SinglePolyModel(LstSqModel):
    degree: int
    
    def __init__(self, x, y, n: int) -> None:
        super().__init__(x, y)
        self.degree = n
    
    
    def transX(self):
        self.x = np.vstack((np.zeros_like(self.x), np.power(self.x, self.degree + 1))).T
        return self.x
    
        
    def transY(self):
        self.y = self.y
        return self.y
    
    
    def fit_transform(self):
        super().fit_transform()
        self.w = self.w[1]
        return self.w
    
'''
    The model y = a e^{bx}, which return the parameter w = (a, b)
'''
class ExpModel(LstSqModel):
    def __init__(self, x, y) -> None:
        super().__init__(x, y, 2)
    
    def transX(self):
        self.x = self.x
        return self.x
    
    def transY(self):
        self.y = np.log(self.y)
        return self.y
        
    def fit_transform(self):
        super().fit_transform()
        self.w[0] = np.exp(self.w[0])
        self.w[1] = np.exp(self.w[1])
        return self.w
    
'''
    The model with y = a ln x + b
'''
class LogModel(LstSqModel):
    def transX(self):
        x1 = np.ones_like(self.x)
        x2 = np.log(self.x)
        self.x = np.vstack((x1, x2)).T
        return self.x
    
    
    def transY(self):
        self.y = self.y
        return self.y 
