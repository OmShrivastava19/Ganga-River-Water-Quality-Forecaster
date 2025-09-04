# Ganga River Water Quality Real-time Forecast Dashboard

An AI-powered water quality forecasting system for the Ganga River using advanced LSTM neural networks, real-time weather integration, and intelligent analysis through Google's Gemini AI.

## Overview

This dashboard provides 3-day forecasts for critical water quality parameters across major cities along the Ganga River. The system combines historical water quality data with real-time weather conditions to predict Dissolved Oxygen (DO), Biochemical Oxygen Demand (BOD), and Fecal Coliform levels.

## Features

### Core Functionality
- **LSTM Neural Network Forecasting**: Advanced time series prediction for water quality parameters
- **Multi-City Coverage**: Monitoring stations from Rishikesh to Kolkata
- **Real-time Weather Integration**: OpenWeather API for current conditions and forecasts
- **AI-Powered Analysis**: Google Gemini AI for intelligent water quality assessment
- **Interactive Web Interface**: User-friendly Gradio-based dashboard

### Key Parameters Monitored
- **Dissolved Oxygen (DO)**: Critical for aquatic ecosystem health (mg/L)
- **Biochemical Oxygen Demand (BOD)**: Indicator of organic pollution levels (mg/L)
- **Fecal Coliform**: Bacterial contamination measurement (MPN/100ml)

### Covered Cities
- Rishikesh
- Haridwar
- Kanpur
- Prayagraj (Allahabad)
- Varanasi
- Patna
- Kolkata

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**:
```bash
git clone https://github.com/OmShrivastava19/Ganga-River-Water-Quality-Forecaster
cd Ganga-River-Water-Quality-Forecaster
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (create `.env` file):
```env
WEATHER_API_KEY=your_openweather_api_key_here
Maps_API_KEY=your_google_maps_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

4. **Run the application**:
```bash
python app.py
```

5. **Access the dashboard**:
Open your browser and navigate to `http://127.0.0.1:7860`

## API Configuration

### OpenWeather API
- **Purpose**: Real-time weather data and 5-day forecasts
- **Sign up**: [OpenWeatherMap](https://openweathermap.org/api)
- **Free tier**: 1,000 calls/day

### Google Maps API
- **Purpose**: Geocoding and location services
- **Setup**: [Google Cloud Console](https://console.cloud.google.com/)
- **Enable**: Geocoding API
- **Billing**: Required (but free tier available)

### Google Gemini API
- **Purpose**: AI-powered water quality analysis
- **Access**: [Google AI Studio](https://aistudio.google.com/)
- **Note**: Currently in preview/beta

## Technical Architecture

### Machine Learning Pipeline
1. **Data Preprocessing**: Historical data cleaning and feature engineering
2. **Sequence Generation**: Time series sequences for LSTM training
3. **Model Training**: 64-unit LSTM with dropout regularization
4. **Forecasting**: 3-day ahead predictions with confidence intervals

### System Components
- **FirebaseDataManager**: Data retrieval and processing
- **WeatherDataManager**: Real-time weather integration
- **GoogleMapsManager**: Location services and mapping
- **GeminiAIAnalyzer**: AI-powered water quality assessment
- **LSTMForecaster**: Neural network prediction engine
- **DashboardVisualizer**: Plot generation and visualization
- **WaterQualityDashboard**: Main orchestration class

## Usage

### Basic Operation
1. Select a city from the dropdown menu
2. Click "Generate Forecast Dashboard"
3. View the three forecast plots:
   - Dissolved Oxygen trends
   - BOD predictions
   - Fecal Coliform forecasts
4. Review weather information and AI analysis

### Advanced Features
- **Interactive Plots**: Hover for detailed values
- **Weather Impact Analysis**: Understanding meteorological effects on water quality
- **AI Recommendations**: Actionable insights for water quality management
- **Historical Trends**: 30-day historical context for predictions

## Data Sources and Methodology

### Historical Data
- **Source**: Central Pollution Control Board (CPCB) portal
- **Parameters**: DO, BOD, Fecal Coliform, Temperature, Precipitation, River Discharge
- **Frequency**: Daily measurements
- **Coverage**: 90-day historical window for training

### Forecasting Model
- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Input Features**: 6 parameters over 7-day sequences
- **Output**: 3-day forecasts for water quality parameters
- **Validation**: 80/20 train-validation split with early stopping

### Weather Integration
- **Current Conditions**: Temperature, humidity, pressure, wind speed
- **Forecast Data**: 5-day weather predictions
- **Impact Analysis**: Weather effects on water quality parameters

## Fallback Mechanisms

The system includes robust fallback options:
- **Mock Weather Data**: Used when OpenWeather API is unavailable
- **Simulated Coordinates**: Default locations when Google Maps API fails
- **Rule-based Analysis**: Basic water quality assessment when Gemini AI is unavailable
- **Synthetic Historical Data**: Generated data for demonstration when database is offline

## File Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # API keys (create this file)
└── .gitignore           # Git ignore rules
```

### Code Structure
- Follow PEP 8 style guidelines
- Add docstrings for all classes and methods
- Include error handling for all API calls
- Write unit tests for new functionality

## Limitations and Considerations

### Model Limitations
- Predictions are based on historical patterns and may not capture unprecedented events
- Model accuracy depends on data quality and completeness
- 3-day forecast horizon balances accuracy and utility

### API Dependencies
- Requires active internet connection for real-time features
- API rate limits may affect high-frequency usage
- Third-party service availability impacts functionality

### Data Considerations
- Historical data may have gaps or inconsistencies
- Weather-water quality correlations vary by location and season
- Model performance may degrade without regular retraining

## Troubleshooting

### Common Issues

**TensorFlow Installation Issues**:
```bash
pip install tensorflow==2.8.0
```

**Gradio Interface Not Loading**:
- Check port 7860 is not in use
- Try different port: `interface.launch(server_port=7861)`

**API Errors**:
- Verify API keys in `.env` file
- Check API quotas and billing status
- Review network connectivity

**Memory Issues**:
- Reduce LSTM model size in `LSTMForecaster` class
- Decrease historical data window
- Use CPU-only TensorFlow if GPU memory is limited

## License

This project is intended for educational and research purposes. Please ensure compliance with data sources' terms of service and API usage policies.

## Acknowledgments

- Central Pollution Control Board (CPCB) for water quality data standards
- OpenWeatherMap for weather data services
- Google for AI and mapping services
- TensorFlow and scikit-learn communities for machine learning tools
- Gradio team for the web interface framework

## Contact

For questions, issues, or contributions, please refer to the project repository.