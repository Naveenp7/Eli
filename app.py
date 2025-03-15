from flask import Flask, request, jsonify, send_from_directory, redirect, session, render_template, send_file
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import sqlite3
import os
from functools import wraps
import secrets
import requests
import google.generativeai as genai
from google.api_core.exceptions import NotFound, ResourceExhausted
from dotenv import load_dotenv
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = secrets.token_hex(32)

# Get API keys from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

# Validate API keys
if not gemini_api_key or not weather_api_key:
    logger.error("API keys are not set. Check your .env file.")
    raise ValueError("API keys are not set. Please check your .env file.")

logger.debug(f"Gemini API Key: {gemini_api_key[:10]}... (partial)")
logger.debug(f"Weather API Key: {weather_api_key}")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# Dynamically select a supported, non-deprecated model
preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro']  # Prioritize these models
available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

# Filter models to exclude deprecated ones and prioritize preferred models
selected_model_name = None
for model in available_models:
    if 'deprecated' in str(model).lower():  # Skip deprecated models
        continue
    if model.name in preferred_models:
        selected_model_name = model.name
        break
if not selected_model_name:  # Fallback to first non-deprecated model
    for model in available_models:
        if 'deprecated' not in str(model).lower():
            selected_model_name = model.name
            break
if not selected_model_name:
    logger.error("No non-deprecated models support generateContent. Check API access or version.")
    raise ValueError("No non-deprecated models support generateContent. Please check your API access.")

gemini_model = genai.GenerativeModel(selected_model_name)
logger.debug(f"Selected Gemini model: {selected_model_name}")

DATABASE = 'users.db'

def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise Exception(f"Database connection failed: {str(e)}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return redirect('/dashboard' if 'username' in session else '/login')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return send_file('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return redirect('/login')
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password, details) VALUES (?, ?, ?)',
                       (username, password, ''))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            return jsonify({'message': 'Login successful', 'token': secrets.token_hex(32), 'redirect': '/dashboard'}), 200
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/update', methods=['POST'])
@login_required
def update():
    data = request.json
    details = data.get('details')
    if not details:
        return jsonify({'error': 'Details are required'}), 400
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET details = ? WHERE username = ?', (details, session['username']))
        conn.commit()
        return jsonify({'message': 'Details updated successfully'}), 200
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/appliances', methods=['GET'])
@login_required
def get_appliances():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM appliances WHERE username = ? ORDER BY bill_date DESC", (session['username'],))
        appliances = [dict(row) for row in cursor.fetchall()]
        return jsonify(appliances)
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/submit-data', methods=['POST'])
@login_required
def submit_data():
    data = request.get_json()
    institution_type = data.get('institutionType')
    bill_date = data.get('billDate')
    historical_units = data.get('historicalUnits')
    previous_bill = data.get('previousBill')
    appliances = data.get('appliances')

    if not all([institution_type, bill_date, historical_units, previous_bill, appliances]) or not isinstance(appliances, list):
        return jsonify({'error': 'All fields are required'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for appliance in appliances:
            cursor.execute(
                'INSERT INTO appliances (username, institution_type, bill_date, historical_units, previous_bill, appliance_name, avg_hours_used) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (session['username'], institution_type, bill_date, historical_units, previous_bill, appliance.get('appliance'), appliance.get('hoursUsed'))
            )
        conn.commit()
        return jsonify({'status': 'success'})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/insights', methods=['GET'])
@login_required
def get_insights():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT institution_type, 
                   SUM(historical_units) as total_units, 
                   AVG(previous_bill) as avg_bill, 
                   COUNT(DISTINCT appliance_name) as total_appliances 
            FROM appliances 
            WHERE username = ? 
            GROUP BY institution_type
        """, (session['username'],))
        insights = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT appliance_name, 
                   SUM(avg_hours_used) as total_hours, 
                   COUNT(*) as usage_count
            FROM appliances 
            WHERE username = ?
            GROUP BY appliance_name
        """, (session['username'],))
        appliance_usage = [dict(row) for row in cursor.fetchall()]

        return jsonify({'summary': insights, 'appliance_usage': appliance_usage})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/appliance-insights', methods=['GET'])
@login_required
def appliance_insights():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT appliance_name, 
                   SUM(avg_hours_used) as total_hours
            FROM appliances
            WHERE username = ?
            GROUP BY appliance_name
        """, (session['username'],))
        appliance_data = [dict(row) for row in cursor.fetchall()]
        return jsonify(appliance_data)
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/recommendations', methods=['GET'])
@login_required
def get_recommendations():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT appliance_name, SUM(avg_hours_used) as total_hours
            FROM appliances
            WHERE username = ?
            GROUP BY appliance_name
        """, (session['username'],))
        usage = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not usage:
            return jsonify({'recommendations': 'No usage data available to provide recommendations.'})

        # Fetch weather for dynamic recommendation
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q=Edappal&appid={weather_api_key}&units=metric"
        logger.debug(f"Fetching weather for recommendations with URL: {weather_url}")
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        logger.debug(f"Weather data for recommendations: {weather_data}")

        if 'main' not in weather_data or 'temp' not in weather_data['main']:
            raise KeyError("Weather data missing 'main' or 'temp' fields")
        temperature = weather_data['main']['temp']

        if not weather_data.get('weather', []):
            raise KeyError("Weather data missing 'weather' array")
        description = weather_data['weather'][0]['description']

        prompt = f"Given this appliance usage data: {usage}, and current temperature in Edappal: {temperature}°C, suggest ways to reduce electricity costs. Provide specific, actionable recommendations."
        logger.debug(f"Gemini prompt: {prompt}")
        response = gemini_model.generate_content(prompt)
        logger.debug(f"Gemini response: {response}")
        recommendations = response.text
        return jsonify({'recommendations': recommendations})
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed in /recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Weather API request failed: {str(e)}'}), 500
    except NotFound as e:
        logger.error(f"Gemini model not found or unsupported: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Gemini model error: {str(e)}'}), 500
    except ResourceExhausted as e:
        logger.error(f"Gemini API quota exceeded: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Gemini API quota exceeded: {str(e)}'}), 429
    except KeyError as e:
        logger.error(f"Invalid weather data format: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Invalid weather data format: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/anomalies', methods=['GET'])
@login_required
def detect_anomalies():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bill_date, historical_units, previous_bill
            FROM appliances
            WHERE username = ?
            ORDER BY bill_date ASC
        """, (session['username'],))
        data = cursor.fetchall()
        conn.close()

        if len(data) < 2:
            return jsonify({'anomalies': 'Insufficient data for anomaly detection. At least 2 entries are required.'})

        X = np.array([[row['historical_units'], row['previous_bill']] for row in data])
        model = IsolationForest(contamination=0.1)
        anomalies = model.fit_predict(X)
        anomaly_indices = [i for i, pred in enumerate(anomalies) if pred == -1]

        anomaly_list = [
            {'bill_date': data[i]['bill_date'], 'units': data[i]['historical_units'], 'bill': data[i]['previous_bill']}
            for i in anomaly_indices
        ]
        return jsonify({'anomalies': anomaly_list})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Anomaly detection error: {str(e)}'}), 500

@app.route('/forecast', methods=['GET'])
@login_required
def forecast_bill():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bill_date, previous_bill
            FROM appliances
            WHERE username = ?
            ORDER BY bill_date ASC
        """, (session['username'],))
        data = cursor.fetchall()
        conn.close()

        if len(data) < 3:
            return jsonify({'error': 'Insufficient data for forecasting. At least 3 entries are required.'}), 400

        bills = [row['previous_bill'] for row in data]
        model = ARIMA(bills, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        forecasted_bill = float(forecast[0])
        return jsonify({'forecasted_bill': forecasted_bill})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Forecasting error: {str(e)}'}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bill_date, historical_units, previous_bill, appliance_name, avg_hours_used
            FROM appliances
            WHERE username = ?
            ORDER BY bill_date DESC
            LIMIT 5
        """, (session.get('username', ''),))
        recent_data = [dict(row) for row in cursor.fetchall()]
        conn.close()

        prompt = f"User's recent electricity usage data: {recent_data}\nUser's question: {user_message}\nProvide a detailed and helpful response based on the data."
        logger.debug(f"Gemini prompt: {prompt}")
        response = gemini_model.generate_content(prompt)
        logger.debug(f"Gemini response: {response}")
        bot_reply = response.text
        return jsonify({"response": bot_reply})
    except NotFound as e:
        logger.error(f"Gemini model not found or unsupported: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Gemini model error: {str(e)}'}), 500
    except ResourceExhausted as e:
        logger.error(f"Gemini API quota exceeded: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Gemini API quota exceeded: {str(e)}'}), 429
    except Exception as e:
        logger.error(f"Unexpected error in /chatbot: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/efficiency', methods=['GET'])
@login_required
def appliance_efficiency():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT appliance_name, avg_hours_used, historical_units
            FROM appliances
            WHERE username = ?
        """, (session['username'],))
        data = cursor.fetchall()
        conn.close()

        if not data:
            return jsonify({'efficiency': 'No data available for efficiency analysis.'})

        power_ratings = {
            "Fan": 75,
            "Light Bulb": 60,
            "AC": 1500,
            "Fridge": 200,
            "Iron": 1000,
            "Electric Stove": 2000,
            "Television": 100,
            "Oven": 1200,
            "Mixer/Blender": 500,
            "Washing Machine": 500,
            "Vacuum Cleaner": 800
        }

        efficiency_report = {}
        total_entries = len(data)
        for row in data:
            appliance = row['appliance_name']
            if appliance in power_ratings:
                expected_units = (power_ratings[appliance] * row['avg_hours_used'] * 30) / 1000
                actual_units = row['historical_units'] / total_entries
                efficiency_report[appliance] = {
                    'expected_units': round(expected_units, 2),
                    'actual_units': round(actual_units, 2),
                    'efficiency': 'Efficient' if actual_units <= expected_units else 'Inefficient - Consider upgrading'
                }

        return jsonify({'efficiency': efficiency_report})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Efficiency analysis error: {str(e)}'}), 500

@app.route('/weather', methods=['GET'])
@login_required
def get_weather():
    try:
        city = request.args.get('city', default='Edappal', type=str)
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        logger.debug(f"Fetching weather for {city} with URL: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        logger.debug(f"Weather data received: {weather_data}")

        if 'main' not in weather_data or 'temp' not in weather_data['main']:
            raise KeyError("Weather data missing 'main' or 'temp' fields")
        temperature = weather_data['main']['temp']

        if not weather_data.get('weather', []):
            raise KeyError("Weather data missing 'weather' array")
        description = weather_data['weather'][0]['description']

        return jsonify({
            'city': city,
            'temperature': temperature,
            'description': description,
            'message': f'Current weather in {city}: {temperature}°C, {description}'
        })
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Weather API request failed: {str(e)}'}), 500
    except KeyError as e:
        logger.error(f"Invalid weather data format: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Invalid weather data format: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in /weather: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict', methods=['GET'])
@login_required
def predict_bill():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bill_date, historical_units, previous_bill
            FROM appliances
            WHERE username = ?
            ORDER BY bill_date ASC
        """, (session['username'],))
        data = cursor.fetchall()

        if not data:
            return jsonify({'error': 'No historical data available for this user'}), 400

        usage = np.array([row['historical_units'] for row in data if row['historical_units'] is not None]).reshape(-1, 1)
        bills = np.array([row['previous_bill'] for row in data if row['previous_bill'] is not None])

        if len(usage) < 2 or len(bills) < 2:
            return jsonify({'error': 'Insufficient data points for prediction (minimum 2 required)'}), 400

        model = LinearRegression()
        model.fit(usage, bills)

        future_usage = request.args.get('future_usage', type=float)
        if future_usage is None:
            return jsonify({'error': 'Future usage parameter is required'}), 400
        if future_usage <= 0:
            return jsonify({'error': 'Future usage must be a positive value'}), 400

        prediction = model.predict([[future_usage]])
        return jsonify({'predicted_bill': float(prediction[0])})
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)