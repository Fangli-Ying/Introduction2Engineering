#!/usr/bin/env python
# coding: utf-8

# # Tensor Basics

# Numpy to tensor and some basics

# In[1]:


import torch


# In[2]:


x =  torch.empty(2,3)
x


# In[3]:


y= torch.rand(2,3,4)
y


# In[4]:


z = torch.ones(2,2)
z


# In[5]:


m = torch.ones(2,2, dtype= torch.double)#float16
m


# In[6]:


print(m.size())


# In[7]:


h= torch.tensor([1.2, 1.])
h


# In[8]:


type(h)


# In[9]:


x= torch.rand(2,3)
y= torch.rand(2,3)
x


# In[10]:


y


# In[11]:


z= x+y
z


# In[12]:


z= torch.add(x,y)
z


# In[13]:


y


# In[14]:


#inplace add with add_
y.add_(x)
y


# In[15]:


#same for sub
z1= torch.sub(x,y)
z1


# In[16]:


z=x-y
z


# In[17]:


y


# In[18]:


y.sub_(x)
y


# ### same for mul - * and div - / 

# In[19]:


x= torch.rand(5,3)
x


# In[20]:


x[1,:] #1 is the second(0,1,2...) row 行 and ：is all columns 


# In[21]:


x[1,1] # the second row and second column


# In[22]:


x[1,1].item() # get the one element with actual value


# In[23]:


x= torch.rand(4,4)
y= x.view(16)# reshape from 4*4 to 16*1
y


# In[24]:


z= x.view(-1,8)# reshape from 4*4 to any (-1 is decide by itself)* 8
z


# In[25]:


z.size


# In[26]:


z.size()


# ## Numpy to tensor

# In[27]:


a = torch.ones(5)
a


# In[28]:


type(a)


# In[29]:


b= a.numpy()
b


# In[30]:


type(b)


# ## note inplace operation when pointing the same place

# In[31]:


a.add_(1)
a


# In[32]:


b


# ## tensor to numpy

# In[33]:


h= torch.from_numpy(b)
h


# In[34]:


b+=1 #inplace operation from numpy to tensor
b


# In[35]:


h


# In[36]:


#windows
if torch.cuda.is_available():
    device =  torch.device("cuda")
    x= torch.ones(5,device= device)#specify the gpu variable
    y= torch.ones(5)
    y= y.to(device)# or you can define it then move to gpu
    z= x+y
    z = z.to("cpu")
    print(z)


# In[37]:


x= torch.ones(5,requires_grad =True)# need to caculate the gradient


# ## gradient

# In[38]:


x= torch.randn(3)
x


# In[39]:


x= torch.randn(3, requires_grad =True)# need to caculate the gradient
x


# In[40]:


y = x+2


# ![VS](figures/BB.PNG)

# In[41]:


y # we can see backward for add


# In[42]:


z=y*y*2 # we can see backward for multiply
z


# In[43]:


h = z.mean()
h


# In[44]:


h.backward() #dz/dx but it can not go second time and the requires_grad should be True


# In[45]:


x.grad


# ![VS](figures/chain.PNG)

# In[46]:


v =torch.tensor([0.1,1.0,0.1],dtype=torch.float32)
v


# In[47]:


b=y*y

b.backward(v)
x.grad


# ## caculate with no gradient

# In[48]:


x = torch.randn(3, requires_grad=True)
x
#x.requires_grad_(False)
#x.detach()
#with torch.no_grad()


# In[49]:


x.requires_grad_(False)


# In[50]:


x = torch.randn(3, requires_grad=True)
y=x.detach()
y


# In[51]:


x = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y=x+2
y


# In[52]:


x


# ## accumulate the gradient

# In[53]:


w = torch.ones(4, requires_grad=True)
for epoch in range(1):
    model_output =  (w*3).sum()
    model_output.backward()
    
w.grad    
    


# In[54]:


w = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output =  (w*3).sum()
    model_output.backward()
    
w.grad  


# In[55]:


w = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output =  (w*3).sum()
    model_output.backward()
    
w.grad  


# In[56]:


w = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output =  (w*3).sum()
    model_output.backward()
    print(w.grad )
    w.grad.zero_()
 


# In[57]:


#opt = torch.optim.SGD(w,lr=0.01)
#opt.step()
#opt.zero_grad()
#we should also prevent the optimizer sumup the gradient too


# ![VS](figures/cr.PNG)

# ![VS](figures/crbp.PNG)

# ![VS](figures/crbp1.PNG)

# In[58]:


import torch
x= torch.tensor(1.0)
y= torch.tensor(2.0)

w= torch.tensor(1.0, requires_grad = True)

#foward pass and compute loss
y_hat = w*x
loss = (y_hat-y)**2
loss


# In[59]:


#backward pass
loss.backward()
w.grad


# In[60]:


#update weight


# ![VS](figures/man.PNG)

# In[61]:


import numpy as np

#f=w*x
# =2*x
#X= np.array([1,2,3,4],dtype=np.float32)
#Y= np.array([2,4,6,8],dtype=np.float32)
X= torch.tensor([1,2,3,4],dtype=torch.float32)
Y= torch.tensor([2,4,6,8],dtype=torch.float32)
w =0.0

#model prediction

def forward(x):
     return w*x


#loss =MSE

def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

     



#gradient
#MSE = 1/N*(w*x-y)**2
#dJ/dw = 1/N 2x(w*x-y)

def gradient(x,y,y_prediected):
    return np.dot(2*x, y_pred-y).mean()
a = forward(5)
print(f'Prediection before training:f(5) = {a:.8f}')


# In[62]:




#Training
learning_rate=0.1
n_iters =10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)
    
    #loss
    l= loss(Y,y_pred)
    
    #gradients
    dw = gradient(X,Y,y_pred)
    
    # update weights
    w-=learning_rate*dw
    
    if epoch%1 == 0:
        print(f'epoch {epoch+1}:w = {w:.3f}, loss ={l:.8f}')
        
print(f'Prediection before training:f(5) = {forward(5):.3f}')

