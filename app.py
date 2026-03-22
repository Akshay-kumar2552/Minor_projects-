import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#00FFAA;'>🚦 Traffic Prediction System</h1>", unsafe_allow_html=True)

# Load data
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Preprocessing
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.dayofweek
df = df.drop(['date_time', 'weather_description'], axis=1)

# Create traffic level
def traffic_level(x):
    if x < 2000:
        return "Low"
    elif x < 4000:
        return "Medium"
    else:
        return "High"

df['traffic_level'] = df['traffic_volume'].apply(traffic_level)

# Encoding
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

df['weather_main'] = le_weather.fit_transform(df['weather_main'])
df['traffic_level'] = le_traffic.fit_transform(df['traffic_level'])  

# Model
X = df[['temp','rain_1h','snow_1h','clouds_all','weather_main','hour','day']]
y = df['traffic_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI
st.sidebar.header("Enter Details")

temp = st.sidebar.number_input("Temperature")
rain = st.sidebar.number_input("Rain (last hour)")
snow = st.sidebar.number_input("Snow (last hour)")
clouds = st.sidebar.slider("Cloud %", 0, 100)
weather = st.sidebar.selectbox("Weather", ["Clouds","Clear","Rain","Mist","Snow"])
hour = st.sidebar.slider("Hour", 0, 23)
day = st.sidebar.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
day = day_map[day]
predict_btn = st.sidebar.button("🚀 Predict Traffic")

if predict_btn:
    weather_encoded = le_weather.transform([weather])[0]

    prediction = model.predict([[temp, rain, snow, clouds, weather_encoded, hour, day]])

    result = le_traffic.inverse_transform(prediction)

    st.markdown(f"""
    <div style='background-color:#1ABC9C; padding:20px; border-radius:10px; text-align:center;'>
        <h2 style='color:white;'>🚗 Predicted Traffic Level: {result[0]}</h2>
    </div>
    """, unsafe_allow_html=True)