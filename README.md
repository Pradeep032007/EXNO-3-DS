# EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="377" height="438" alt="image" src="https://github.com/user-attachments/assets/22077443-c581-4e86-8d68-a11d261c2db5" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="174" height="240" alt="image" src="https://github.com/user-attachments/assets/97ada51c-2ef5-412a-b627-1fb42c114457" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="417" height="449" alt="image" src="https://github.com/user-attachments/assets/77a625fc-c747-4318-b2a7-a8013723b6f2" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="416" height="438" alt="image" src="https://github.com/user-attachments/assets/9e2258e7-45c2-4444-8c3c-e7476296d203" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

one = OneHotEncoder(sparse_output=False)   # <-- use sparse_output
df2 = df.copy()
enc = pd.DataFrame(one.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```

<img width="527" height="444" alt="image" src="https://github.com/user-attachments/assets/32578960-8be9-41de-bfb6-f7b6fe7df419" />


```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="806" height="444" alt="image" src="https://github.com/user-attachments/assets/c8adc9d5-a439-4f4a-9dc8-24e143dca03a" />


```
pip install --upgrade category_encoders
```
<img width="1364" height="442" alt="image" src="https://github.com/user-attachments/assets/a1c661c7-a4ed-4bf2-a50d-0cb5c1ca46a7" />


```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="580" height="447" alt="image" src="https://github.com/user-attachments/assets/7ba86570-d2cc-4e83-8073-a4573a76e61f" />



```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="581" height="453" alt="image" src="https://github.com/user-attachments/assets/7f5e7f39-5d17-4c9e-9913-28dd84117034" />


```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="852" height="467" alt="image" src="https://github.com/user-attachments/assets/34985183-ffa3-40ff-a59e-162af45b5795" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="665" height="441" alt="image" src="https://github.com/user-attachments/assets/70131451-dee5-45b1-a3d9-dad2e511ee48" />


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="951" height="518" alt="image" src="https://github.com/user-attachments/assets/a12ff8c6-1778-4837-923e-48d909020c4a" />

```
df.skew()
```

<img width="358" height="251" alt="image" src="https://github.com/user-attachments/assets/d8867bf9-654c-4b5e-b6df-41d9eca56670" />


```
np.log(df["Highly Positive Skew"])
```

<img width="314" height="563" alt="image" src="https://github.com/user-attachments/assets/e91e3ac8-b7b3-4aa2-abca-b51129b6cb7e" />


```
np.sqrt(df["Highly Positive Skew"])
```

<img width="358" height="562" alt="image" src="https://github.com/user-attachments/assets/841b8ddc-3ff3-435d-b419-dc15a8309de8" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="339" height="565" alt="image" src="https://github.com/user-attachments/assets/6a1b4e62-b14a-4ccb-b1ee-683d664b5959" />


```
np.square(df["Highly Positive Skew"])
```

<img width="412" height="561" alt="image" src="https://github.com/user-attachments/assets/c2564fac-5748-4fa4-b84d-74f6327fa77c" />



```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1331" height="528" alt="image" src="https://github.com/user-attachments/assets/60a4d410-7072-4340-9775-3f6ab29695f2" />


```
df.skew()
```


<img width="405" height="302" alt="image" src="https://github.com/user-attachments/assets/4e51fc4d-6a44-4729-b453-7c2b90787090" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="453" height="339" alt="image" src="https://github.com/user-attachments/assets/eaaa1a1f-b288-4058-8399-2e8afc49aeaf" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


<img width="1336" height="583" alt="image" src="https://github.com/user-attachments/assets/a2287015-2fe0-41e1-bfef-fae10d639461" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="721" height="537" alt="image" src="https://github.com/user-attachments/assets/af9654fc-3734-42ea-93fd-559bda848157" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```


<img width="716" height="542" alt="image" src="https://github.com/user-attachments/assets/d335cb73-0417-4291-8802-5ae0d56f7d48" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="707" height="544" alt="image" src="https://github.com/user-attachments/assets/af07a3b6-91dd-4189-9c42-7e2bdeeadd96" />




```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


<img width="703" height="539" alt="image" src="https://github.com/user-attachments/assets/ad2e2fab-7fc6-4e5e-9926-82e1fa788904" />


```
dt=pd.read_csv("data.csv")
dt
```

<img width="583" height="446" alt="image" src="https://github.com/user-attachments/assets/2af6672b-ff78-4103-998d-b2d89707c007" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Ord_1"]=qt.fit_transform(dt[["Target"]])
sm.qqplot(dt['Target'],line='45')
plt.show()

```


<img width="1309" height="600" alt="image" src="https://github.com/user-attachments/assets/aa0581b7-cdbb-41ca-b945-012dc9b95f49" />





```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


<img width="710" height="543" alt="image" src="https://github.com/user-attachments/assets/3ad9d9e6-9c14-4058-bdbf-4ba10e6163f1" />


# RESULT:

      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
