"""
Ganga River Water Quality Real-time Forecast Dashboard
AI-powered forecasting system with interactive Gradio interface
Enhanced with Google Maps, OpenWeather API, and Gemini AI
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import requests
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure API keys with proper error handling
GOOGLE_MAPS_API_KEY = os.getenv('Maps_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Print API configuration status
print("\n🔑 API Configuration Status:")
if WEATHER_API_KEY:
    print("✅ OpenWeather API configured")
else:
    print("⚠️ OpenWeather API key not configured - using mock weather data")

if GOOGLE_MAPS_API_KEY:
    print("✅ Google Maps API key configured")
else:
    print("⚠️ Google Maps API key not configured - mapping features disabled")

if GEMINI_API_KEY:
    print("✅ Gemini API key configured")
else:
    print("⚠️ Gemini API key not configured - using mock AI analysis")

# Configure matplotlib for better plots
plt.style.use('default')  # Changed from 'seaborn-v0_8' which may not be available
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class FirebaseDataManager:
    """Manages Firebase data retrieval and processing"""
    
    def __init__(self, credentials_path: str = None, database_url: str = None):
        """Initialize Firebase connection"""
        self.use_mock_data = True  # Default to mock data for demo
        print("Using mock data for demonstration")
    
    def get_data_from_firebase(self, city: str) -> pd.DataFrame:
        """Retrieve historical data from Firebase for a given city"""
        return self._generate_mock_data(city)
    
    def _generate_mock_data(self, city: str) -> pd.DataFrame:
        """Generate realistic mock data for demonstration"""
        print(f"Generating mock data for {city}")
        
        # Generate 90 days of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base values for different cities
        base_values = {
            "Rishikesh": {"do": 7.5, "bod": 2.0, "fecal_coliform": 200},
            "Haridwar": {"do": 7.0, "bod": 2.5, "fecal_coliform": 300},
            "Kanpur": {"do": 5.5, "bod": 4.0, "fecal_coliform": 800},
            "Prayagraj": {"do": 6.0, "bod": 3.5, "fecal_coliform": 600},
            "Varanasi": {"do": 5.0, "bod": 4.5, "fecal_coliform": 1000},
            "Patna": {"do": 4.5, "bod": 5.0, "fecal_coliform": 1200},
            "Kolkata": {"do": 4.0, "bod": 5.5, "fecal_coliform": 1500}
        }
        
        base = base_values.get(city, {"do": 6.0, "bod": 3.0, "fecal_coliform": 500})
        
        # Generate realistic time series with trends and seasonality
        np.random.seed(hash(city) % 2**32)  # Consistent seed based on city name
        
        # Add seasonal patterns (weekly and monthly cycles)
        weekly_cycle = np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        monthly_cycle = np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        
        # Generate data with realistic variations
        do_values = base["do"] + 0.5 * weekly_cycle + 0.3 * monthly_cycle + np.random.normal(0, 0.3, len(dates))
        bod_values = base["bod"] + 0.3 * weekly_cycle + 0.2 * monthly_cycle + np.random.normal(0, 0.2, len(dates))
        fc_values = base["fecal_coliform"] + 100 * weekly_cycle + 50 * monthly_cycle + np.random.normal(0, 50, len(dates))
        
        # Ensure values are within realistic ranges
        do_values = np.clip(do_values, 3.0, 10.0)
        bod_values = np.clip(bod_values, 1.0, 8.0)
        fc_values = np.clip(fc_values, 50, 2000)
        
        # Generate weather and discharge data
        temperature = 25 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates))
        precipitation = np.random.exponential(5, len(dates))
        discharge = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 30) + np.random.normal(0, 50, len(dates))
        
        df = pd.DataFrame({
            'do': do_values,
            'bod': bod_values,
            'fecal_coliform': fc_values,
            'temperature': temperature,
            'precipitation': precipitation,
            'discharge': discharge
        }, index=dates)
        
        return df

class WeatherDataManager:
    """Manages real-time weather data integration using OpenWeather API"""
    
    def __init__(self, api_key: str = None):
        """Initialize weather data manager"""
        self.api_key = api_key or WEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            print("⚠️ Weather API key not configured - will use mock weather data")
    
    def get_weather_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Fetch real-time weather data for specific coordinates"""
        if not self.api_key:
            return self._get_mock_weather_data(latitude, longitude)
        
        try:
            # Get current weather
            current_url = f"{self.base_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(current_url, params=params, timeout=10)
            response.raise_for_status()
            
            current_data = response.json()
            
            # Get 5-day forecast
            forecast_url = f"{self.base_url}/forecast"
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            
            # Extract and format weather data
            weather_info = {
                'current': {
                    'temperature': current_data['main']['temp'],
                    'feels_like': current_data['main']['feels_like'],
                    'humidity': current_data['main']['humidity'],
                    'pressure': current_data['main']['pressure'],
                    'wind_speed': current_data['wind']['speed'],
                    'wind_direction': current_data['wind'].get('deg', 0),
                    'description': current_data['weather'][0]['description'],
                    'icon': current_data['weather'][0]['icon'],
                    'timestamp': datetime.now().isoformat()
                },
                'forecast': self._process_forecast_data(forecast_data),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'city': current_data.get('name', 'Unknown'),
                    'country': current_data.get('sys', {}).get('country', 'Unknown')
                }
            }
            
            print(f"✅ Weather data fetched successfully for {weather_info['location']['city']}")
            return weather_info
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {e}")
            return self._get_mock_weather_data(latitude, longitude)
    
    def get_weather_data_by_city(self, city_name: str) -> Dict[str, Any]:
        """Get weather data by city name"""
        city_coordinates = {
            "Rishikesh": (30.0869, 78.2676),
            "Haridwar": (29.9457, 78.1642),
            "Kanpur": (26.4499, 80.3319),
            "Prayagraj": (25.4358, 81.8463),
            "Varanasi": (25.3176, 82.9739),
            "Patna": (25.5941, 85.1376),
            "Kolkata": (22.5726, 88.3639)
        }
        
        if city_name in city_coordinates:
            lat, lon = city_coordinates[city_name]
            return self.get_weather_data(lat, lon)
        else:
            print(f"⚠️ Unknown city: {city_name}, using default coordinates")
            return self.get_weather_data(25.3176, 82.9739)
    
    def _process_forecast_data(self, forecast_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process 5-day forecast data into daily summaries"""
        daily_forecasts = []
        daily_data = {}
        
        for item in forecast_data['list']:
            date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(item)
        
        # Calculate daily averages
        for date, items in daily_data.items():
            temps = [item['main']['temp'] for item in items]
            humidities = [item['main']['humidity'] for item in items]
            wind_speeds = [item['wind']['speed'] for item in items]
            precipitations = [item.get('rain', {}).get('3h', 0) for item in items]
            
            daily_forecasts.append({
                'date': date,
                'temperature': round(sum(temps) / len(temps), 1),
                'humidity': round(sum(humidities) / len(humidities), 1),
                'wind_speed': round(sum(wind_speeds) / len(wind_speeds), 1),
                'precipitation': round(sum(precipitations), 1),
                'description': items[0]['weather'][0]['description']
            })
        
        return daily_forecasts[:5]
    
    def _get_mock_weather_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Generate mock weather data when API is unavailable"""
        print(f"📝 Generating mock weather data for coordinates ({latitude}, {longitude})")
        
        base_temp = 25 + 10 * np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        
        return {
            'current': {
                'temperature': round(base_temp + np.random.normal(0, 2), 1),
                'feels_like': round(base_temp + np.random.normal(0, 2), 1),
                'humidity': round(max(30, min(90, 60 + np.random.normal(0, 15)))),
                'pressure': round(1013 + np.random.normal(0, 10)),
                'wind_speed': round(max(0, 5 + np.random.exponential(2)), 1),
                'wind_direction': round(np.random.uniform(0, 360)),
                'description': 'partly cloudy',
                'icon': '02d',
                'timestamp': datetime.now().isoformat()
            },
            'forecast': [
                {
                    'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'temperature': round(base_temp + np.random.normal(0, 3), 1),
                    'humidity': round(max(30, min(90, 60 + np.random.normal(0, 20)))),
                    'wind_speed': round(max(0, 5 + np.random.exponential(2)), 1),
                    'precipitation': round(max(0, np.random.exponential(2)), 1),
                    'description': 'partly cloudy'
                } for i in range(1, 6)
            ],
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'city': 'Mock City',
                'country': 'Mock Country'
            }
        }

class GoogleMapsManager:
    """Manages Google Maps API integration for geocoding and mapping"""
    
    def __init__(self, api_key: str = None):
        """Initialize Google Maps manager"""
        self.api_key = api_key or GOOGLE_MAPS_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                # Use requests instead of googlemaps library to avoid dependency issues
                self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
                print("✅ Google Maps API configured")
            except Exception as e:
                print(f"❌ Failed to configure Google Maps API: {e}")
        else:
            print("⚠️ Google Maps API key not configured - mapping features disabled")
    
    def get_coordinates(self, location_name: str) -> Dict[str, Any]:
        """Get coordinates for a location name using Google Maps Geocoding API"""
        if not self.api_key:
            return self._get_mock_coordinates(location_name)
        
        try:
            params = {
                'address': location_name,
                'key': self.api_key
            }
            
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'OK' or not data['results']:
                print(f"⚠️ No coordinates found for: {location_name}")
                return self._get_mock_coordinates(location_name)
            
            location = data['results'][0]
            geometry = location['geometry']
            
            lat = geometry['location']['lat']
            lng = geometry['location']['lng']
            
            result = {
                'coordinates': {
                    'latitude': lat,
                    'longitude': lng
                },
                'address': {
                    'formatted': location['formatted_address'],
                    'city': 'Unknown',
                    'state': 'Unknown',
                    'country': 'India'
                },
                'place_id': location['place_id'],
                'types': location.get('types', []),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Coordinates found for {location_name}: ({lat}, {lng})")
            return result
            
        except Exception as e:
            print(f"❌ Error in geocoding for {location_name}: {e}")
            return self._get_mock_coordinates(location_name)
    
    def _get_mock_coordinates(self, location_name: str) -> Dict[str, Any]:
        """Generate mock coordinates when API is unavailable"""
        print(f"📝 Generating mock coordinates for: {location_name}")
        
        mock_coordinates = {
            "Har Ki Pauri, Haridwar": (29.9457, 78.1642),
            "Dashashwamedh Ghat, Varanasi": (25.3176, 82.9739),
            "Manikarnika Ghat, Varanasi": (25.3176, 82.9739),
            "Assi Ghat, Varanasi": (25.3176, 82.9739),
            "Sangam, Prayagraj": (25.4358, 81.8463),
            "Adi Ganga, Kolkata": (22.5726, 88.3639),
            "Ganga Barrage, Kanpur": (26.4499, 80.3319)
        }
        
        # Try to find a match
        for key, coords in mock_coordinates.items():
            if any(word.lower() in location_name.lower() for word in key.split()):
                lat, lng = coords
                return {
                    'coordinates': {
                        'latitude': lat,
                        'longitude': lng
                    },
                    'address': {
                        'formatted': f"{location_name}, India",
                        'city': 'Mock City',
                        'state': 'Mock State',
                        'country': 'India'
                    },
                    'place_id': 'mock_place_id',
                    'types': ['mock_location'],
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Mock coordinates - API unavailable'
                }
        
        # Default coordinates (Varanasi)
        return {
            'coordinates': {
                'latitude': 25.3176,
                'longitude': 82.9739
            },
            'address': {
                'formatted': f"{location_name}, Varanasi, India",
                'city': 'Varanasi',
                'state': 'Uttar Pradesh',
                'country': 'India'
            },
            'place_id': 'mock_place_id',
            'types': ['mock_location'],
            'timestamp': datetime.now().isoformat(),
            'note': 'Default mock coordinates - API unavailable'
        }

class GeminiAIAnalyzer:
    """AI-powered water quality analysis using Google's Gemini API"""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini AI analyzer"""
        self.api_key = api_key or GEMINI_API_KEY
        self.model = None
        
        if self.api_key:
            try:
                # Use requests directly instead of google.generativeai library
                self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                print("✅ Gemini AI configured")
            except Exception as e:
                print(f"❌ Failed to configure Gemini AI: {e}")
        else:
            print("⚠️ Gemini API key not configured - using mock AI analysis")
    
    def get_water_quality_summary(self, data_dict: Dict[str, Any]) -> str:
        """Generate a human-readable water quality summary using Gemini AI"""
        if not self.api_key:
            return self._get_mock_water_quality_summary(data_dict)
        
        try:
            prompt = f"""
            As a water quality expert, analyze the following water quality data and provide:
            
            1. **Interpretation**: Rate the water quality based on standard parameters
            2. **Summary**: 2-3 sentence plain-language summary
            3. **Advisory**: Simple recommendation
            
            Water Quality Data:
            Location: {data_dict.get('location', 'Unknown')}
            pH: {data_dict.get('ph', 'N/A')}
            BOD: {data_dict.get('bod', 'N/A')} mg/L
            Dissolved Oxygen: {data_dict.get('dissolved_oxygen', 'N/A')} mg/L
            Turbidity: {data_dict.get('turbidity', 'N/A')} NTU
            Fecal Coliform: {data_dict.get('fecal_coliform', 'N/A')} MPN/100ml
            
            Please provide a concise, professional analysis suitable for environmental monitoring reports.
            """
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            data = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }]
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print("❌ No valid response from Gemini API")
                    return self._get_mock_water_quality_summary(data_dict)
            else:
                print(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return self._get_mock_water_quality_summary(data_dict)
            
        except Exception as e:
            print(f"❌ Error generating AI water quality summary: {e}")
            return self._get_mock_water_quality_summary(data_dict)
    
    def _get_mock_water_quality_summary(self, data_dict: Dict[str, Any]) -> str:
        """Generate mock water quality summary when AI is unavailable"""
        location = data_dict.get('location', 'Unknown Location')
        ph = data_dict.get('ph', 0)
        bod = data_dict.get('bod', 0)
        do = data_dict.get('dissolved_oxygen', 0)
        turbidity = data_dict.get('turbidity', 0)
        fecal_coliform = data_dict.get('fecal_coliform', 0)
        
        # Basic interpretation logic
        ph_status = "neutral" if 6.5 <= ph <= 8.5 else "outside normal range"
        do_status = "adequate" if do >= 4.0 else "low"
        bod_status = "acceptable" if bod <= 30 else "high"
        
        # Generate advisory
        if do < 4.0 or bod > 30 or fecal_coliform > 1000:
            advisory = "⚠️ Water is NOT suitable for bathing or drinking. Avoid contact."
        elif do < 6.0 or bod > 15:
            advisory = "⚠️ Water quality is marginal. Limited recreational use recommended."
        else:
            advisory = "✅ Water quality is acceptable for most recreational activities."
        
        return f"""
Water Quality Summary for {location}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Interpretation:**
- pH Level: {ph_status} (pH: {ph})
- Dissolved Oxygen: {do_status} ({do} mg/L)
- BOD: {bod_status} ({bod} mg/L)
- Turbidity: {turbidity} NTU
- Fecal Coliform: {fecal_coliform} MPN/100ml

**Summary:**
The water quality at {location} shows {ph_status} pH levels with {do_status} dissolved oxygen content. 
Biochemical oxygen demand is {bod_status}, indicating the level of organic pollution present.

**Advisory:**
{advisory}

Note: This is a basic analysis. AI service unavailable - using rule-based assessment.
"""

class LSTMForecaster:
    """LSTM-based forecasting model for water quality prediction"""
    
    def __init__(self, n_steps: int = 7, n_forecast: int = 3):
        """Initialize the LSTM forecaster"""
        self.n_steps = n_steps
        self.n_forecast = n_forecast
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_names = ['do', 'bod', 'fecal_coliform', 'temperature', 'precipitation', 'discharge']
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.n_steps - self.n_forecast + 1):
            X.append(data[i:(i + self.n_steps)])
            output_sequence = data[(i + self.n_steps):(i + self.n_steps + self.n_forecast), :3]
            y.append(output_sequence.flatten())
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.n_forecast * 3)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the LSTM model on historical data"""
        try:
            # Select features
            features = df[self.feature_names].values
            
            # Scale the data
            scaled_features = self.scaler.fit_transform(features)
            
            # Prepare sequences
            X, y = self.prepare_sequences(scaled_features)
            
            if len(X) < 10:
                raise ValueError("Insufficient data for training")
            
            # Split data (80% train, 20% validation)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build and train model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,  # Reduced epochs for faster training
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make forecast
            last_sequence = scaled_features[-self.n_steps:]
            forecast = self._make_forecast(last_sequence)
            
            # Calculate metrics
            val_predictions = self.model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, val_predictions)
            mae = mean_absolute_error(y_val, val_predictions)
            
            return {
                'forecast': forecast,
                'history': history.history,
                'metrics': {'mse': mse, 'mae': mae},
                'last_sequence': last_sequence
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def _make_forecast(self, last_sequence: np.ndarray) -> np.ndarray:
        """Make forecast using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Reshape for prediction
        X_pred = last_sequence.reshape(1, self.n_steps, len(self.feature_names))
        
        # Make prediction
        pred_scaled = self.model.predict(X_pred, verbose=0)
        
        # Reshape prediction to (n_forecast, 3)
        pred_reshaped = pred_scaled.reshape(self.n_forecast, 3)
        
        # Create dummy array for inverse scaling
        dummy_array = np.zeros((self.n_forecast, len(self.feature_names)))
        dummy_array[:, :3] = pred_reshaped
        
        # Inverse transform
        pred_unscaled = self.scaler.inverse_transform(dummy_array)
        
        return pred_unscaled[:, :3]

class DashboardVisualizer:
    """Creates visualizations for the dashboard"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.colors = {
            'historical': '#2E86AB',
            'forecast': '#A23B72',
            'background': '#F8F9FA'
        }
    
    def create_forecast_plots(self, df: pd.DataFrame, forecast: np.ndarray, 
                            city: str, n_forecast: int = 3) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Create forecast plots for DO, BOD, and Fecal Coliform"""
        
        # Get last 30 days of historical data
        historical_data = df.tail(30)
        
        # Create forecast dates
        last_date = historical_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=n_forecast, freq='D')
        
        # Create plots
        fig1 = self._create_do_plot(historical_data, forecast_dates, forecast[:, 0], city)
        fig2 = self._create_bod_plot(historical_data, forecast_dates, forecast[:, 1], city)
        fig3 = self._create_fc_plot(historical_data, forecast_dates, forecast[:, 2], city)
        
        return fig1, fig2, fig3
    
    def _create_do_plot(self, historical: pd.DataFrame, forecast_dates: pd.DatetimeIndex, 
                       forecast_values: np.ndarray, city: str) -> plt.Figure:
        """Create Dissolved Oxygen forecast plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical.index, historical['do'], 
               color=self.colors['historical'], linewidth=2, label='Historical DO')
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_values, 
               color=self.colors['forecast'], linewidth=2, linestyle='--', 
               label='Forecast DO')
        
        # Add confidence interval
        ax.fill_between(forecast_dates, 
                       forecast_values * 0.9, 
                       forecast_values * 1.1, 
                       alpha=0.3, color=self.colors['forecast'])
        
        ax.set_title(f'Dissolved Oxygen (DO) Forecast - {city}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('DO (mg/L)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _create_bod_plot(self, historical: pd.DataFrame, forecast_dates: pd.DatetimeIndex, 
                        forecast_values: np.ndarray, city: str) -> plt.Figure:
        """Create BOD forecast plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical.index, historical['bod'], 
               color=self.colors['historical'], linewidth=2, label='Historical BOD')
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_values, 
               color=self.colors['forecast'], linewidth=2, linestyle='--', 
               label='Forecast BOD')
        
        # Add confidence interval
        ax.fill_between(forecast_dates, 
                       forecast_values * 0.9, 
                       forecast_values * 1.1, 
                       alpha=0.3, color=self.colors['forecast'])
        
        ax.set_title(f'Biochemical Oxygen Demand (BOD) Forecast - {city}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('BOD (mg/L)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _create_fc_plot(self, historical: pd.DataFrame, forecast_dates: pd.DatetimeIndex, 
                       forecast_values: np.ndarray, city: str) -> plt.Figure:
        """Create Fecal Coliform forecast plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical.index, historical['fecal_coliform'], 
               color=self.colors['historical'], linewidth=2, label='Historical Fecal Coliform')
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_values, 
               color=self.colors['forecast'], linewidth=2, linestyle='--', 
               label='Forecast Fecal Coliform')
        
        # Add confidence interval
        ax.fill_between(forecast_dates, 
                       forecast_values * 0.9, 
                       forecast_values * 1.1, 
                       alpha=0.3, color=self.colors['forecast'])
        
        ax.set_title(f'Fecal Coliform Forecast - {city}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Fecal Coliform (MPN/100ml)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig

class WaterQualityDashboard:
    """Main dashboard class that orchestrates the entire application"""
    
    def __init__(self):
        """Initialize the dashboard components"""
        self.firebase_manager = FirebaseDataManager()
        self.weather_manager = WeatherDataManager()
        self.maps_manager = GoogleMapsManager()
        self.ai_analyzer = GeminiAIAnalyzer()
        self.forecaster = LSTMForecaster()
        self.visualizer = DashboardVisualizer()
        self.cities = ["Rishikesh", "Haridwar", "Kanpur", "Prayagraj", "Varanasi", "Patna", "Kolkata"]
    
    def generate_forecast_dashboard(self, city: str) -> Tuple[plt.Figure, plt.Figure, plt.Figure, str, str]:
        """Generate the complete forecast dashboard for a city with weather and AI analysis"""
        try:
            print(f"Generating forecast for {city}...")
            
            # Get historical data
            df = self.firebase_manager.get_data_from_firebase(city)
            
            if df.empty:
                raise ValueError(f"No data available for {city}")
            
            # Train model and get forecast
            result = self.forecaster.train_model(df)
            
            if result is None:
                raise ValueError("Failed to train forecasting model")
            
            forecast = result['forecast']
            
            # Create visualizations
            fig1, fig2, fig3 = self.visualizer.create_forecast_plots(
                df, forecast, city, self.forecaster.n_forecast
            )
            
            # Get weather data
            weather_data = self.weather_manager.get_weather_data_by_city(city)
            weather_summary = self._format_weather_summary(weather_data)
            
            # Get AI analysis
            current_data = {
                'location': city,
                'ph': 7.5,  # Mock pH value
                'bod': float(df['bod'].iloc[-1]),
                'dissolved_oxygen': float(df['do'].iloc[-1]),
                'turbidity': 25,  # Mock turbidity
                'fecal_coliform': float(df['fecal_coliform'].iloc[-1])
            }
            ai_analysis = self.ai_analyzer.get_water_quality_summary(current_data)
            
            print(f"Forecast generated successfully for {city}")
            return fig1, fig2, fig3, weather_summary, ai_analysis
            
        except Exception as e:
            print(f"Error generating forecast for {city}: {e}")
            # Return empty plots and error messages
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'Error generating forecast for {city}\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Error - {city}', fontsize=16)
            
            error_msg = f"Error generating data for {city}: {str(e)}"
            return fig, fig, fig, error_msg, error_msg
    
    def _format_weather_summary(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into a readable summary"""
        if not weather_data or 'current' not in weather_data:
            return "Weather data unavailable"
        
        current = weather_data['current']
        location = weather_data.get('location', {})
        
        summary = f"""
**Current Weather for {location.get('city', 'Unknown City')}**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🌡️ Temperature: {current.get('temperature', 'N/A')}°C (Feels like {current.get('feels_like', 'N/A')}°C)
💧 Humidity: {current.get('humidity', 'N/A')}%
🌪️ Wind: {current.get('wind_speed', 'N/A')} m/s
📊 Pressure: {current.get('pressure', 'N/A')} hPa
☁️ Conditions: {current.get('description', 'N/A').title()}

**Impact on Water Quality:**
- Temperature affects dissolved oxygen levels
- High humidity may indicate increased precipitation
- Wind speed influences water circulation and aeration
        """
        
        return summary.strip()

def create_enhanced_interface():
    """Create an enhanced Gradio interface with multiple outputs"""
    
    # Initialize dashboard
    dashboard = WaterQualityDashboard()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
    }
    .plot-container {
        margin: 20px 0;
    }
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Ganga Water Quality Dashboard") as interface:
        gr.HTML("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1>🌊 Ganga River Water Quality Real-time Forecast Dashboard</h1>
            <h3>AI-Powered Water Quality Forecasting System</h3>
            <p>Advanced LSTM neural networks with real-time weather integration and AI analysis</p>
            <hr style='margin: 20px 0;'>
        </div>
        """)
        
        with gr.Row():
            city_dropdown = gr.Dropdown(
                choices=dashboard.cities,
                label="🏙️ Select City Along Ganga River",
                value="Kanpur",
                info="Choose a monitoring station to view water quality forecasts and analysis"
            )
            
            generate_btn = gr.Button("🚀 Generate Forecast Dashboard", variant="primary", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Water Quality Forecasts (3-Day Prediction)</h3>")
                
                do_plot = gr.Plot(label="Dissolved Oxygen (DO) Forecast", show_label=True)
                bod_plot = gr.Plot(label="Biochemical Oxygen Demand (BOD) Forecast", show_label=True)
                fc_plot = gr.Plot(label="Fecal Coliform Forecast", show_label=True)
            
            with gr.Column(scale=1):
                gr.HTML("<h3>🌤️ Weather Information</h3>")
                weather_info = gr.Markdown(
                    value="Select a city and click 'Generate Forecast Dashboard' to view weather data",
                    label="Current Weather & Impact Analysis"
                )
                
                gr.HTML("<h3>🤖 AI Water Quality Analysis</h3>")
                ai_analysis = gr.Markdown(
                    value="Select a city and click 'Generate Forecast Dashboard' to view AI analysis",
                    label="AI-Powered Quality Assessment"
                )
        
        # Information section
        with gr.Accordion("ℹ️ About This Dashboard", open=False):
            gr.Markdown("""
            ### System Overview
            This dashboard uses advanced machine learning techniques to forecast water quality parameters for the Ganga River:
            
            **🧠 AI Technologies Used:**
            - **LSTM Neural Networks**: For time series forecasting of water quality parameters
            - **Google Gemini AI**: For intelligent water quality analysis and recommendations
            - **OpenWeather API**: For real-time weather data integration
            - **Google Maps API**: For location services and mapping
            
            **📊 Key Parameters Monitored:**
            - **Dissolved Oxygen (DO)**: Critical for aquatic life health
            - **Biochemical Oxygen Demand (BOD)**: Indicator of organic pollution
            - **Fecal Coliform**: Measure of bacterial contamination
            
            **🌍 Coverage Areas:**
            Major cities along the Ganga River from Rishikesh to Kolkata
            
            **⚡ Real-time Features:**
            - Live weather data integration
            - AI-powered water quality assessment
            - Interactive forecasting visualizations
            - Location-based monitoring station mapping
            """)
        
        # Event handlers
        generate_btn.click(
            fn=dashboard.generate_forecast_dashboard,
            inputs=[city_dropdown],
            outputs=[do_plot, bod_plot, fc_plot, weather_info, ai_analysis]
        )
        
        # Auto-generate on city selection
        city_dropdown.change(
            fn=dashboard.generate_forecast_dashboard,
            inputs=[city_dropdown],
            outputs=[do_plot, bod_plot, fc_plot, weather_info, ai_analysis]
        )
        
        # Examples section
        gr.Examples(
            examples=[
                ["Kanpur"],
                ["Varanasi"],
                ["Patna"],
                ["Kolkata"],
                ["Haridwar"]
            ],
            inputs=[city_dropdown],
            label="🔍 Quick Examples - Click to Test Different Cities"
        )
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
            <p><strong>Ganga River Water Quality Monitoring System</strong></p>
            <p>Powered by TensorFlow, Scikit-learn, OpenWeather API, Google Maps API, and Gemini AI</p>
            <p><em>For research and environmental monitoring purposes</em></p>
        </div>
        """)
    
    return interface

def main():
    """Launch the enhanced Gradio application"""
    
    print("🚀 Initializing Ganga River Water Quality Forecast Dashboard...")
    print("📊 Setting up AI models and API integrations...")
    
    # Create and launch the enhanced interface
    interface = create_enhanced_interface()
    
    print("✅ Dashboard initialized successfully!")
    print("🌐 Launching web interface...")
    
    # For Beanstalk deployment, use different settings
    interface.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=8080,       # Beanstalk expects port 8080
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False  # Don't auto-open browser in cloud
    )
    
if __name__ == "__main__":
    main()  