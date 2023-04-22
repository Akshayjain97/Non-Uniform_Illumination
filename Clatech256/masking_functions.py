import numpy as np


def one(k):
   u,v=32,32
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def two(k):
   u,v=32,32
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def three(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def four(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def five(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=abs(16-x)*abs(16-y)
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
 
def six(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           a=50-abs(16-x)*abs(16-y)
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def seven(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
           if(x<=16 and y>16):a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
           if(x>16 and y<=16):a=-((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
           if(x>16 and y>16):a=-(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def eight(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
           if(x<=16 and y>16):a=-(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
           if(x>16 and y<=16):a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
           if(x>16 and y>16):a=-(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
def nine(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):
       for y in range(0,32):
           a=0
           if(0<=y<=5 or 10<=y<=15 or 20<=y<=25 or 30<=y<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
           else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
 
def ten(k):
   mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
   u,v=32,32
   for x in range(0,32):	
       for y in range(0,32):
           a=0
           if(0<=x<=5 or 10<=x<=15 or 20<=x<=25 or 30<=x<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
           else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
           a=k*a
           mask[x][y]=np.array((a,a,a))
           
   return mask
   
  
  

