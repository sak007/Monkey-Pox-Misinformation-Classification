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
data2.to_csv(csvname_output,index='False')  

