import pandas as pd
import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import datetime

pd.options.mode.chained_assignment = None
nltk.download('stopwords')
csvname_input=r"C:\Users\rarun\Desktop\monkeypox-followup.csv"
csvname_output=r"C:\Users\rarun\Desktop\monkeypox-followup_preprocessed.csv"
data=pd.read_csv(csvname_input)
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
data2=data.copy()
rows=len(data)

day_tweeted=[]
month_tweeted=[]
year_tweeted=[]
hour_tweeted=[]
minute_tweeted=[]

day_created=[]
month_created=[]
year_created=[]
hour_created=[]
minute_created=[]

for i in range (0,rows):

    #Preprocess the tweets
    oldtext=data2['text'][i]
    #removes hashtags
    newtext=' '.join(re.sub("(#[A-Za-z0-9]+)"," ",str(oldtext)).split())
    #removes UserID
    newtext=' '.join(re.sub("(@[A-Za-z0-9]+)"," ",str(newtext)).split())
    #removes urls
    newtext=' '.join(re.sub("(_URL_)"," ",str(newtext)).split())
    #removes additional urls as well
    newtext=' '.join(re.sub("(\w+:\/\/\S+)"," ",str(newtext)).split())
    #keeps only alphanumeric characters
    newtext= re.sub('[^a-zA-Z0-9]'," ",str(newtext))
    #converts to lower case and splits
    newtext=newtext.lower()
    newtext=newtext.split()
    #Performs Stemming    
    ps=PorterStemmer()
    newtext= [ps.stem(word) for word in newtext if word not in set(stopwords.words('english'))]
    newtext=' '.join(newtext)
    data2['text'][i]=newtext

    #Preprocess the user description
    oldtext=data['user description'][i]
    #removes hashtags
    newtext=' '.join(re.sub("(#[A-Za-z0-9]+)"," ",str(oldtext)).split())
    #removes UserID
    newtext=' '.join(re.sub("(@[A-Za-z0-9]+)"," ",str(newtext)).split())
    #removes urls
    newtext=' '.join(re.sub("(_URL_)"," ",str(newtext)).split())
    #removes additional urls as well
    newtext=' '.join(re.sub("(\w+:\/\/\S+)"," ",str(newtext)).split())
    #keeps only alphanumeric characters
    newtext= re.sub('[^a-zA-Z0-9]'," ",str(newtext))
    #converts to lower case and splits
    newtext=newtext.lower()
    newtext=newtext.split()
    #Performs Stemming
    ps=PorterStemmer()
    newtext= [ps.stem(word) for word in newtext if word not in set(stopwords.words('english'))]
    newtext=' '.join(newtext)
    data2['user description'][i]=newtext

    #Gets the day, month, year, hour, minutes of the tweet
    date_time = data2['created_at'][i] 
    q=date_time.split("/")
    day,month=q[0],q[1].lstrip("0")
    temp = str(q.pop()).split(" ")
    year, hour, minute = temp[0], str(temp[1]).split(":")[0], str(temp[1]).split(":")[1]
    day_tweeted.append(day)
    month_tweeted.append(month)
    year_tweeted.append(year)
    hour_tweeted.append(hour)
    minute_tweeted.append(minute)

    #Gets the day, month, year, hour, minutes of the account creation
    date_time = data2['user created at'][i] 
    q=date_time.split("/")
    day,month=q[0],q[1].lstrip("0")
    temp = str(q.pop()).split(" ")
    year, hour, minute = temp[0], str(temp[1]).split(":")[0], str(temp[1]).split(":")[1]
    day_created.append(day)
    month_created.append(month)
    year_created.append(year)
    hour_created.append(hour)
    minute_created.append(minute)
    
data2['user has url'] = data2['user has url'].replace({False: 0})
data2['user has url'] = data2['user has url'].replace({True: 1})
data2['user is verified'] = data2['user is verified'].replace({False: 0})
data2['user is verified'] = data2['user is verified'].replace({True: 1})
data2['beto_flag'] = data2['beto_flag'].replace({False: 0})
data2['beto_flag'] = data2['beto_flag'].replace({True: 1})      

data2["day_tweeted"]= day_tweeted
data2["month_tweeted"]= month_tweeted
data2["year_tweeted"]= year_tweeted
data2["hour_tweeted"]= hour_tweeted
data2["minute_tweeted"]= minute_tweeted

data2["day_created"]=day_created
data2["month_created"]=month_created
data2["year_created"]=year_created
data2["hour_created"]=hour_created
data2["minute_created"]=minute_created

#Saving changes to the csv file
#data2.to_csv(csvname_output,index='False')

corpus=[]
y=[]
import numpy as np
for i in range(rows):
    newtext=data2['text'][i]
    if(len(str((newtext)))!=0 and str(newtext)!='nan'):
        corpus.append(newtext)
        y.append(data2['binary_class'][i])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X2 = cv.fit_transform(corpus).toarray()
y = np.array(y)

from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size = 0.2, random_state = 0)


from sklearn.svm import SVC
from sklearn import svm
SVC2 = SVC(kernel = 'linear', random_state = 0)
SVC2.fit(X2_train, y2_train)

from sklearn.metrics import accuracy_score
y_pred=SVC2.predict(X2_test)
print("The Accuracy using Support Vector Clustering on Data-set 2: ",accuracy_score(y2_test,y_pred))

