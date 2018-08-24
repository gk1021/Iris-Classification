import tensorflow
import tflearn 
import matplotlib.pyplot as plt
import numpy as np
from tflearn.data_utils import load_csv

ignore=[0,1] 
#,columns_to_ignore=ignore
data,labels=load_csv('iris.csv',target_column=4,categorical_labels=True,n_classes=3,columns_to_ignore=ignore)

net=tflearn.input_data(shape=[None,2])
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,16)
#net=tflearn.fully_connected(net,32)

net=tflearn.fully_connected(net,3,activation='softmax')
net=tflearn.regression(net)
x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
 
print("labels",data)
model=tflearn.DNN(net)
model.fit(data,labels,n_epoch=50,batch_size=8,show_metric=True, validation_set=0.1)
#pred=model.predict(flowers)
index=0
d1=[]
d2=[]
for i in data:
  j=0
  
  while j<2:
    d1.append(float(i[j]))
    d2.append(float(i[j+1]))
    j=j+2
d3=np.column_stack((d1,d2))
#print(d3)
pred=model.predict(d3)
i=0
while i<150:
  if pred[i][0]>=pred[i][1] and pred[i][0]>=pred[i][2]:
    
    #print("setosa")
    x1.append(d3[i][0])
    y1.append(d3[i][1])
     
  elif pred[i][1]>=pred[i][2]:
    
    #print("versicolor")
    x2.append(d3[i][0])
    y2.append(d3[i][1])
     
    
  else:
    
    #print("virginica") 
    x3.append(d3[i][0])
    y3.append(d3[i][1]) 
  i=i+1  
  
plt.scatter(x1,y1,c="red",s=10,label='Setosa' )
plt.scatter(x2,y2,c="green",s=10,label='versicolor')
plt.scatter(x3,y3,c="blue",s=10,label='virginica') 
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.legend()
plt.show()   