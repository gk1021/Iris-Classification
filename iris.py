import matplotlib.pyplot as plt
import pandas as pd  


df = pd.read_csv('iris.csv')
df = df[['petal_length', 'species']]

rows, columns = df.shape

x = df[df.columns[0]]
label = df[df.columns[1]]
#print(rows)
#print(label)
setosa = 0.0
versicolor = 0.0
virginica =0.0
sn=0
vc=0
vn=0

 #Loop begin
i=0
while i<rows:
  if label[i]=='setosa':
    setosa=setosa+x[i]
    sn=sn+1
  elif label[i]=='versicolor':
    versicolor=versicolor+x[i]
    vc=vc+1
  elif label[i]=='virginica':
    virginica=virginica+x[i]
    vn=vn+1 
  i = i+1
 
#Loop end
#Average
setosa_avg = setosa/sn
versicolor_avg = versicolor/vc
virginica_avg = virginica/vn

print(setosa_avg)
print(versicolor_avg)
print(virginica_avg) 


#print(virginica)
  

#plt.hist(df)
#plt.show()
