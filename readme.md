# World Trends Analyzer

A powerful, interactive Streamlit dashboard for analyzing global socioeconomic, health, and environmental trends from 1990 to 2022. Built with Python, it leverages World Bank and other public datasets to provide insights through visualizations, machine learning, time-series forecasting, and AI-driven analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The World Trends Analyzer enables users to explore global trends across metrics like GDP per capita, literacy rates, renewable energy consumption, CO₂ emissions, child mortality, and immunization coverage. It features interactive visualizations, clustering, forecasting, and AI insights, making it a valuable tool for researchers, policymakers, and data enthusiasts.

## Features
- **Overview**: Interactive choropleth maps (Plotly), global metrics (e.g., average GDP, HDI, CO₂ emissions), and top/bottom 5 countries by HDI.
- **Detailed Analysis**: Heatmaps, yearly/country-wise trends, vaccine coverage, box plots, and KMeans clustering for country segmentation.
- **Country Profiles**: Country-specific trends, distributions, and PCA-based visualizations.
- **Comparison**: Multi-country metric comparisons and HDI rankings.
- **Future Predictions**: Time-series forecasting using Prophet with MAE/RMSE metrics and AI-generated insights via Ollama.
- **AI Insights**: Natural language summaries of trends using Ollama's local LLM models.

## Tech Stack
- **Core**: Python 3.10+
- **Data Processing**: Pandas, NumPy, Dask (for large-scale data merging)
- **Visualization**: Matplotlib, Seaborn, Plotly Express
- **Machine Learning**: Scikit-learn (KMeans, PCA, StandardScaler, LinearRegression), XGBoost
- **Forecasting**: Prophet
- **AI**: Ollama (local LLM for insights)
- **Web Framework**: Streamlit
- **Environment**: Jupyter Notebooks (for data cleaning, EDA, and ML prototyping)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/world-trends-analyzer.git
   cd world-trends-analyzer
   ```
2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv healthenv
   source healthenv/bin/activate  # Linux/Mac
   healthenv\Scripts\activate     # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install Ollama for AI Features**:
   - Follow [Ollama's official guide](https://ollama.ai/) to install Ollama.
   - Pull required models:
     ```bash
     ollama pull data_analyst
     ollama pull future_predict
     ```
5. **Download Datasets**:
   - Place `master_dataset_again.csv`, `HDI_Comparison.csv`, and `Renewable_energy_consumption.csv` in the project root.
   - See [Data Requirements](#data-requirements) for details.

6. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Data Requirements
- **master_dataset_again.csv**: Merged dataset with columns for Country, Year, Vaccine, Observation Value, Life Expectancy, Literacy Rate, Renewable Energy, GDP, CO₂ Emissions, and Child Mortality.
- **HDI_Comparison.csv**: Contains Human Development Index (HDI) data by country.
- **Renewable_energy_consumption.csv**: Country-wise renewable energy data with country codes.
- **Source**: World Bank APIs and other public datasets (cleaned in `Health_data_cleaning.ipynb`).
- **Note**: Ensure datasets are in the project root. Missing data may cause errors in visualizations or forecasts.

## Usage
1. Launch the app (`streamlit run app.py`).
2. Navigate through tabs:
   - **Overview**: Select metrics and years for global choropleth maps and view summary metrics.
   - **Detailed Analysis**: Choose chart types (e.g., heatmap, trends, clusters) and customize with countries or metrics.
   - **Country Profiles**: Analyze specific countries with line plots and box plots.
   - **Comparison**: Compare metrics across multiple countries.
   - **Future Predictions**: Select global or country-specific forecasting, adjust prediction years, and view AI insights.
   - **AI**: Input prompts or use default data for AI-generated trend summaries.
3. **Tips**:
   - Use multiselect for country comparisons.
   - Adjust forecast periods (1-100 years) for predictions.
   - Ensure Ollama is running for AI features (`ollama serve`).

## Project Structure
```
world-trends-analyzer/
├── app.py                    # Main Streamlit application
├── Health_data_cleaning.ipynb # Data merging and cleaning
├── health_ml.ipynb           # ML modeling (e.g., Linear Regression)
├── health_data_Future_prediction.ipynb # Prophet forecasting
├── k_means.ipynb             # KMeans clustering experiments
├── health_data_EDA.ipynb     # Exploratory data analysis
├── requirements.txt          # Python dependencies
├── master_dataset_again.csv  # Main dataset
├── HDI_Comparison.csv        # HDI data
├── Renewable_energy_consumption.csv # Renewable energy data
├── README.md                 # Project documentation
```

## Notebooks
- **Health_data_cleaning.ipynb**: Merges and cleans World Bank datasets using Pandas and Dask, producing `master_dataset_again.csv`.
- **health_ml.ipynb**: Implements Linear Regression for GDP prediction (R²=0.41) and feature analysis.
- **health_data_Future_prediction.ipynb**: Prototypes Prophet-based forecasting for global metrics.
- **k_means.ipynb**: Applies KMeans clustering to GDP vs. Literacy Rate for country segmentation.
- **health_data_EDA.ipynb**: Conducts exploratory analysis with sorted data and trend visualizations.

## Troubleshooting
- **Streamlit Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`).
- **Ollama Issues**: Verify Ollama is running (`ollama serve`) and models (`data_analyst`, `future_predict`) are pulled.
- **Data Errors**: Check dataset files exist and have correct column names (e.g., `Country`, `Year`, `Code`).
- **Memory Issues**: For large datasets, ensure sufficient RAM or use Dask for processing.
- **Plotly Issues**: Update Plotly to 5.24.1 (`pip install plotly==5.24.1`).

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Report bugs or suggest features via GitHub Issues.

## License
MIT License. See `LICENSE` for details.