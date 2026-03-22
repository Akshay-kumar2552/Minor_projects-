import pandas as pd

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Convert date_time
df['date_time'] = pd.to_datetime(df['date_time'])

# Extract features
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.dayofweek

# Drop useless columns
df = df.drop(['date_time', 'weather_description'], axis=1)

print(df.head())

def traffic_level(x):
    if x < 2000:
        return "low"
    elif x < 4000:
        return "medium"
    else:
        return "high"
    
df['traffic_level']=df['traffic_volume'].apply(traffic_level)
print(df[['traffic_level','traffic_volume']].head())

from sklearn.preprocessing import LabelEncoder
le_weather= LabelEncoder()
le_traffic= LabelEncoder()
df['weather_main']=le_weather.fit_transform(df['weather_main'])
df['traffic_level']=le_traffic.fit_transform(df['traffic_level'])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x=df[['temp','rain_1h','snow_1h','clouds_all','weather_main','hour','day']]
y=df['traffic_level']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=RandomForestClassifier()
model.fit(x_train,y_train)
print("Model Trained Successfully ✅")

from sklearn.metrics import accuracy_score
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))

temp=float(input("Enter the temperature:"))
clouds=int(input("Enter the clouds percentage:"))
rain=float(input("Enter the rain (last hour):"))
snow=float(input("Enter the snow (last hour):"))
weather=input("Enter weather(Rain/Clouds/Snow):")
hour=int(input("Enter the time (0-23):"))
day=int(input("Enter the day(0=monday,6=sunday)"))

weather_encoder=le_weather.transform([weather])[0]
prediction=model.predict([[temp,rain,snow,clouds,weather_encoder,hour,day]])

result=le_traffic.inverse_transform(prediction)
print("Pridicted traffic level:",result[0])




 

