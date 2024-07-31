from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction import FeatureHasher
import datetime

app = Flask(__name__)

def extract_date_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def check_holiday_event(selected_date, selected_city):
    selected_date = pd.to_datetime(selected_date)
    holidays_events['date'] = pd.to_datetime(holidays_events['date'])
    selected_day = selected_date.day
    selected_month = selected_date.month
    holidays_events['day'] = holidays_events['date'].dt.day
    holidays_events['month'] = holidays_events['date'].dt.month
    city_match = holidays_events['locale_name'] == selected_city
    day_match = holidays_events['day'] == selected_day
    month_match = holidays_events['month'] == selected_month
    not_transferred = holidays_events['transferred'] == False
    return not holidays_events[day_match & month_match & city_match & not_transferred].empty

stores = pd.read_csv("stores.csv")
holidays_events = pd.read_csv("holidays_events.csv")

pipeline, ref_cols, target = joblib.load('pipelineOne_compressed.pkl')


# Load families.txt
with open('families.txt', 'r') as f:
    families = [line.strip() for line in f]

@app.route('/', methods=['GET', 'POST'])



def index():
    if request.method == 'POST':
        state = request.form.get('state')
        city = request.form.get('city')
        store_type = request.form.get('store_type')
        cluster = request.form.get('cluster')
        store_number = request.form.get('store_number')
        family = request.form.get('family')
        on_promotion = int(request.form.get('on_promotion'))
        transactions = int(request.form.get('transactions'))
        oil_price = float(request.form.get('oil_price'))
        selected_date = request.form.get('selected_date')
        selected_date = pd.to_datetime(selected_date)

        is_holiday = check_holiday_event(selected_date, city)
        pre_holiday_1 = selected_date - pd.Timedelta(days=1)
        pre_holiday_2 = selected_date - pd.Timedelta(days=2)
        is_pre_holiday_1 = check_holiday_event(pre_holiday_1, city)
        is_pre_holiday_2 = check_holiday_event(pre_holiday_2, city)
        is_pre_holiday = False
        if is_holiday:
            is_pre_holiday = False
        elif is_pre_holiday_1 or is_pre_holiday_2:
            is_pre_holiday = True
            is_holiday = False

        selected_date_df = pd.DataFrame({'date': [selected_date]})
        selected_date_df = extract_date_features(selected_date_df)
        day_of_week = selected_date_df['day_of_week'].values[0]
        month = selected_date_df['month'].values[0]
        year = selected_date_df['year'].values[0]

        query = np.array([on_promotion, cluster, transactions, oil_price, is_holiday, is_pre_holiday, day_of_week, month, year, family, city, state, store_type, store_number])
        column_names = ['onpromotion', 'cluster', 'transactions', 'dcoilwtico', 'is_holiday', 'is_pre_holiday', 'day_of_week', 'month', 'year', 'family', 'city', 'state', 'type', 'store_nbr']
        query = pd.DataFrame([query], columns=column_names)
        boolean_columns = ['is_holiday', 'is_pre_holiday']
        for col in boolean_columns:
            if query[col].dtype == 'object':
                query[col] = query[col].map({'True': 1, 'False': 0})
        prediction = pipeline.predict(query)
        predicted_sales = prediction[0]
        return render_template('result.html', predicted_sales=predicted_sales)
    
    states = stores['state'].unique()
    state_city_map = stores.groupby('state')['city'].unique().to_dict()
    city_store_type_map = stores.groupby(['state', 'city'])['type'].unique().to_dict()
    initial_state = states[0]
    cities = state_city_map[initial_state]
    initial_city = cities[0]
    store_types = city_store_type_map[(initial_state, initial_city)]
    clusters = stores[(stores['state'] == initial_state) & (stores['city'] == initial_city) & (stores['type'].isin(store_types))]['cluster'].unique()
    store_numbers = stores[(stores['state'] == initial_state) & (stores['city'] == initial_city) & (stores['type'].isin(store_types)) & (stores['cluster'] == clusters[0])]['store_nbr'].unique()
    
    return render_template('index.html', states=states, cities=cities, store_types=store_types, clusters=clusters, store_numbers=store_numbers, families=families, state_city_map=state_city_map, city_store_type_map=city_store_type_map)

@app.route('/get_cities/<state>')
def get_cities(state):
    print(f"Fetching cities for state: {state}")
    cities = stores[stores['state'] == state]['city'].unique()
    print(f"Found cities: {list(cities)}")
    return jsonify(cities=list(cities))

@app.route('/get_store_types/<state>/<city>')
def get_store_types(state, city):
    print(f"Fetching store types for state: {state}, city: {city}")
    store_types = stores[(stores['state'] == state) & (stores['city'] == city)]['type'].unique()
    print(f"Found store types: {list(store_types)}")
    return jsonify(store_types=list(store_types))

@app.route('/get_clusters/<state>/<city>/<store_type>')
def get_clusters(state, city, store_type):
    print(f"Fetching clusters for state: {state}, city: {city}, store type: {store_type}")
    clusters = stores[(stores['state'] == state) & (stores['city'] == city) & (stores['type'] == store_type)]['cluster'].unique()
    # Convert numpy.int64 to int
    clusters = [int(cluster) for cluster in clusters]
    print(f"Found clusters: {clusters}")
    return jsonify(clusters=clusters)

@app.route('/get_store_numbers/<state>/<city>/<store_type>/<cluster>', methods=['GET'])
def get_store_numbers(state, city, store_type, cluster):
    try:
        # Convert cluster to int for filtering
        cluster = int(cluster)
    except ValueError:
        return jsonify({"error": "Invalid cluster value"}), 400

    print(f"Fetching store numbers for state: {state}, city: {city}, store type: {store_type}, cluster: {cluster}")

    # Filter the DataFrame
    filtered_stores = stores[
        (stores['state'] == state) &
        (stores['city'] == city) &
        (stores['type'] == store_type) &
        (stores['cluster'] == cluster)
    ]

    # Check if any stores match the criteria
    if filtered_stores.empty:
        return jsonify({"message": "No stores found"}), 404

    # Extract store numbers and convert them to integers
    store_numbers = [int(num) for num in filtered_stores['store_nbr'].unique()]

    print(f"Found store numbers: {store_numbers}")

    return jsonify(store_numbers=store_numbers)

if __name__ == '__main__':
    app.run(debug=True)
