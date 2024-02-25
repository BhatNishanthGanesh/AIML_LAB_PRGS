import numpy as np

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100


def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_of_sigmoid(x):
    return x*(1-x)

epoch=1000
learning_rate=0.6
inputlayer_neurons=2
hiddenlayer_neurons=3
outputlayer_neurons=1

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wo=np.random.uniform(size=(hiddenlayer_neurons,outputlayer_neurons))
bo=np.random.uniform(size=(1,outputlayer_neurons))

for i in range(epoch):
    net_h=np.dot(x,wh)+bh
    sigma_h=sigmoid(net_h)
    net_o=np.dot(sigma_h,wo)+bo
    output=sigmoid(net_o)
    
    deltak=(y-output)*derivatives_of_sigmoid(output)
    deltah=deltak.dot(wo.T)*derivatives_of_sigmoid(sigma_h)
    wo=wo+sigma_h.T.dot(deltak)*learning_rate
    wh=wh+x.T.dot(deltah)*learning_rate
    
print(f"Input {x}\n")
print(f"Actual-output: {y}")
print(f"Predicted-output: {output}")
    