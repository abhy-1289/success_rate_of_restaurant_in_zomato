#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('C:/Users/lenovo/OneDrive/Documents/Zomato/zomato.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()
#Returns the number of null values in a column


# In[7]:


feature_na = [feature for feature in df.columns if df[feature].isnull().sum()>0]
feature_na
# returns a list of features having null values


# In[8]:


for feature in feature_na:
    print('{} has {} % of missing values'.format(feature,np.round(df[feature].isnull().sum()/len(df)*100,4)))


# In[9]:


df.info()


# In[10]:


df['approx_cost(for two people)'].isnull()


# In[11]:


df['approx_cost(for two people)'].unique()
# returns all the unique values in the column


# In[12]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))
#As there are commas in the values we have to remove them
#so we convert them into string and replace the comma


# In[13]:


df['approx_cost(for two people)'].unique()


# In[14]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)
#As the commas are removed we change the datatype to float


# In[15]:


df['approx_cost(for two people)'].dtype


# In[16]:


df['rate'].unique()


# In[17]:


df['rate'].isnull().sum()


# In[18]:


# Now we have 4.1/5 we just need 4.1 so we will split and store the fisrt element
# we convert the data element into string and split it
def split(x):
    return x.split('/')[0]


# In[19]:


df['rate'] = df['rate'].astype(str).apply(split)


# In[20]:


df['rate'].unique()


# In[21]:


# we have to replace - and NEW with 0
df['rate'].replace('-',0,inplace=True)
df['rate'].replace('NEW',0,inplace=True)


# In[22]:


df['rate'] = df['rate'].astype(float)


# In[23]:


plt.figure(figsize=(20,12))
df['rest_type'].value_counts().nlargest(20).plot.bar(color='red')
# value counts retuens the no. of times an element occurs
#nlargest() method is used to get n largest values from a data frame or a series


# In[24]:


def sep(x):
    if x in ('Quick Bites','Casual Dining'):
        return 'Quick Bites + Casual Dining'
    else:
        return 'Others'


# In[25]:


df['top_type'] = df['rest_type'].apply(sep)


# In[26]:


df.head()


# In[27]:


import plotly


# In[28]:


import plotly.express as px


# In[29]:


val = df['top_type'].value_counts().values
lab = df['top_type'].value_counts().index


# In[30]:


fig = px.pie(df['rest_type'],names=lab,values=val)
fig.show()


# In[31]:


df.columns


# In[32]:


rest = df.groupby('name').agg({'votes':'sum','url':'count','approx_cost(for two people)':'mean','rate':'mean'}).reset_index()
rest
# We are grouping the restaurents based on no. of votes, branches anf mean of cost and rating


# In[33]:


# Now we are changing the column names for better understanding
rest.columns = ['name','total_votes','total_unities','avg_approx_cost','avg_ratinh']
rest.head()


# In[34]:


rest['votes_per_unity'] = rest['total_votes']/rest['total_unities']
# Adding a new column so that we can get the votes per unity


# In[35]:


popular = rest.sort_values('total_unities',ascending=False)
popular
# sorting the data so we have restaurents with more no. of unities at top


# In[36]:


import seaborn
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20,30))
ax1.text(0.5,0.5, int(popular['total_votes'].mean()),fontsize=50,ha='center')
ax1.text(0.5,0.3,'The avg votes received by restaurants',fontsize=20,ha='center')
ax1.axis('off')
sns.barplot(x='name',y='total_votes',data=popular.sort_values('total_votes',ascending=False).head(5),ax=ax2)
ax2.set_title('Top 5 restaurants with more number of votes',fontsize=30)
sns.barplot(x='name',y='total_votes',data=popular.sort_values('total_votes',ascending=False).query('total_votes > 0').tail(5),ax=ax3)
ax3.set_title('Top 5 restaurants with less number of votes',fontsize=30)


# In[37]:


import seaborn
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20,30))
ax1.text(0.5,0.5, int(popular['avg_approx_cost'].mean()),fontsize=50,ha='center')
ax1.text(0.5,0.3,'The mean approx cost for the restaurants',fontsize=20,ha='center')
ax1.axis('off')
sns.barplot(x='name',y='avg_approx_cost',data=popular.sort_values('avg_approx_cost',ascending=False).head(5),ax=ax2)
ax2.set_title('Top 5 costliest restaurants',fontsize=30)
sns.barplot(x='name',y='avg_approx_cost',data=popular.sort_values('avg_approx_cost',ascending=False).query('avg_approx_cost>0').tail(5),ax=ax3)
ax3.set_title('Top 5 cheapest restaurants',fontsize=30)


# In[38]:


import plotly.express as px
x = df['book_table'].value_counts()
y = ['booking_available','booking_not_available']
fig = px.pie(df['book_table'],names=y,values=x)
fig.show()


# In[39]:


import plotly.express as px
x = df['online_order'].value_counts()
y = ['online_order_available','online_order_not_available']
fig = px.pie(df['online_order'],names=y,values=x)
fig.show()


# In[40]:


# Printing budget friendly restaurants based on the filter given by user
def return_budget(cost,location,rating,restaurant_type):
    filter = (df['approx_cost(for two people)']<=cost) & (df['location']==location) & (df['rate']>=rating) & (df['rest_type']==restaurant_type)
    budget = df[filter]
    return budget['name'].unique()


# In[41]:


return_budget(400,'BTM',4,'Quick Bites')


# In[42]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='app')
locations = pd.DataFrame({'Name':df['location'].unique()})
locations


# In[43]:


lat=[]
lon=[]
for loc in locations['Name']:
    loc = geolocator.geocode(loc)
    if loc is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(loc.latitude)
        lon.append(loc.longitude)


# In[44]:


locations['latitude']=lat
locations['longitude']=lon
locations.head()


# In[45]:


rest_loc=df['location'].value_counts().reset_index()
rest_loc.columns=['Name','count']
restaurant_locations=rest_loc.merge(locations,on='Name').dropna()
restaurant_locations


# In[46]:


import folium
from folium .plugins import HeatMap


# In[47]:


basemap=folium.Map(location=[12.97,77.59])


# In[48]:


HeatMap(data=restaurant_locations[['latitude','longitude','count']]).add_to(basemap)


# In[49]:


basemap


# In[50]:


import wordcloud


# In[51]:


from wordcloud import WordCloud, STOPWORDS


# In[52]:


data=df[df['rest_type']=='Quick Bites']
data


# In[53]:


dishes=''
for word in data['dish_liked'].dropna():
    words=word.split()
    for i in range(len(words)):
        words[i]=words[i].lower()
    dishes=dishes+' '.join(words)+' '


# In[54]:


dishes


# In[55]:


stop_words=set(STOPWORDS)


# In[56]:


word_cloud=WordCloud(stopwords=stop_words,width=1500,height=1500).generate(dishes)


# In[57]:


plt.imshow(word_cloud)
plt.axis('off')


# In[58]:


import re


# In[59]:


dataset=df[df['rest_type']=='Quick Bites']


# In[60]:


total_review=' '
for review in dataset['reviews_list']:
    review=review.lower()
    review=re.sub('[^a-zA-Z]',' ',review)
    review=re.sub('rated',' ',review)
    review=re.sub('x',' ',review)
    review=re.sub(' +',' ',review)
    total_review=total_review+str(review)


# In[61]:


word_cloud2=WordCloud(stopwords=stop_words,width=2500,height=2500).generate(total_review)


# In[62]:


plt.imshow(word_cloud2)
plt.axis('off')


# In[63]:


def assign(x):
    if x>0:
        return 1
    else:
        return 0


# In[64]:


df['rated']=df['rate'].apply(assign)


# In[65]:


df.columns


# In[66]:


new_restaurants=df[df['rated']==0]
train_data=df.query('rated==1')
train_data


# In[67]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[68]:


train_data['target']=train_data['rate'].apply(lambda x:1 if x>=3.75 else 0)


# In[69]:


x=train_data['target'].value_counts()
y=['Not successful','Successful']
fig=px.pie(train_data['target'],names=x,labels=y)
fig.show()


# In[70]:


def count(x):
    return len(x.split(','))


# In[71]:


train_data['tottal_cuisines']=train_data['cuisines'].astype(str).apply(count)
train_data['multiple_types']=train_data['rest_type'].astype(str).apply(count)


# In[72]:


train_data.head()


# In[73]:


train_data.columns


# In[74]:


imp_features=['online_order', 'book_table', 'location', 'rest_type',
       'approx_cost(for two people)',
       'listed_in(type)', 'listed_in(city)', 'target',
       'tottal_cuisines', 'multiple_types']


# In[75]:


new_data=train_data[imp_features]


# In[76]:


new_data


# In[77]:


new_data.isnull().sum()


# In[78]:


new_data.dropna(how='any',inplace=True)


# In[79]:


new_data.shape


# In[80]:


cat_features=[col for col in new_data.columns if new_data[col].dtype=='O']
cat_features


# In[81]:


num_features=[col for col in new_data.columns if new_data[col].dtype!='O']
num_features


# In[82]:


for feature in cat_features:
    print("{} has total {} unique features".format(feature,new_data[feature].nunique()))


# In[83]:


values=new_data['location'].value_counts()/len(new_data)*100
values


# In[84]:


threshold=0.4
imp=values[values>0.4]
imp


# In[85]:


imp.nunique()


# In[86]:


new_data['location']=np.where(new_data['location'].isin(imp.index),new_data['location'],'other')
new_data.nunique()


# In[87]:


values2=new_data['rest_type'].value_counts()/len(new_data)*100
values2


# In[88]:


threshold=1.5
imp2=values2[values2>threshold]
imp2


# In[89]:


len(imp2)


# In[90]:


new_data['rest_type']=np.where(new_data['rest_type'].isin(imp2.index),new_data['rest_type'],'other')
new_data['rest_type'].head()


# In[91]:


for feature in cat_features:
    print("{} has total {} unique features".format(feature,new_data[feature].nunique()))


# In[92]:


data_cat = new_data[cat_features]


# In[93]:


for col in cat_features:
    col_encoded=pd.get_dummies(data_cat[col],prefix=col,drop_first=True)
    data_cat=pd.concat([data_cat,col_encoded],axis=1)
    data_cat.drop(col,axis=1,inplace=True)


# In[94]:


data_cat.head()


# In[95]:


num_features


# In[96]:


data_final=pd.concat([new_data.loc[:,['approx_cost(for two people)', 'target', 'tottal_cuisines', 'multiple_types']],data_cat],axis=1)


# In[97]:


data_final.shape


# In[98]:


X=data_final.drop('target',axis=1)
y=data_final['target']


# In[99]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)


# In[100]:


X_train.shape


# In[101]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)


# In[102]:


predictions=model.predict(X_test)


# In[103]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(predictions,y_test)


# In[104]:


accuracy_score(predictions,y_test)


# In[106]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[107]:


# Inserting all models in a list
models=[]
models.append(('LogisticRegression', LogisticRegression()))
models.append(('Naive Bayes',GaussianNB()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))


# In[108]:


# Printing the confusion matrix and accuracy for each model
for name, model in models:
    print(name)
    model.fit(X_train, y_train)    
    predictions = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions, y_test))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(predictions,y_test))
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




