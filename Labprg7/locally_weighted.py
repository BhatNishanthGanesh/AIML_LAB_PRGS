import numpy as np 
import matplotlib.pyplot as plt

def localregression(xo,x,y,tau):
    xo=[1,xo]
    x=[[1,i] for i in x]
    x=np.asarray(x)
    xw=(x.T)*np.exp(np.sum((x-xo)**2,axis=1)/(-2*tau))
    beta=np.linalg.pinv(xw @x) @xw @y @xo 
    return beta 

def draw(tau):
    prediction=[localregression(xo,x,y,tau) for xo in domain]
    plt.plot(x,y,'o',color='black')
    plt.plot(domain,prediction,color='red')
    plt.show()
    plt.show()
    
x=np.linspace(-3,3,num=1000)
domain=x
y=np.log(np.abs(x**2-1)+.5)
draw(10)
draw(0.1)
draw(0.01)
draw(0.001)
