# ELMO_dashboard

## Overview
This project is designed to visualize article sources and their summaries using Streamlit and Plotly to support ELMO.

## Configuration
The project uses a configuration file to specify the paths to data files and other settings.

Configuration Details:

data_file: Path to the JSON file containing the scrapped articles data.

summaries_file: Path to the JSON file containing the summaries of the articles.

country_origin_file: Path to the JSON file that maps websites to their respective countries and the name of the news outlet.

TIME_SCALE: Time scale for the data visualization. Options include "Daily", "Month", and "Year".

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/akashdrdc/ELMO_dashboard.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
  
3. Run the streamlit application:
    ```bash
    streamlit run app.py
    
## Code Structure
app.py: Main application file containing the Streamlit code.

config.py: Contains configurations to run different datasets.

requirements.txt: List of required Python packages.

README.md: Project documentation.

flags/: Folder containing flag images.

