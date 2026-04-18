# Import Tariffs and Price Regulation: US, China, Vietnam Comparative Study
This repository contains the code for a quantitative comparative analysis of import tariffs, trade openness, and green transition indicators on GDP growth in the United States, China, and Vietnam. The pipeline includes ARDL, GLSAR, Ridge Regression with Bootstrap, and SARIMAX models.

## Installation
Clone the repository:
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/OanhNguyen90/import-tariffs-green-transition-us-china-vietnam).git

cd import-tariffs-green-transition-us-china-vietnam

## Install all required packages using the provided requirements.txt file:
pip install -r requirements.txt

## If you do not have requirements.txt or prefer to install manually, run:
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn pandas-datareader openpyxl
The code requires Python 3.10 or newer.

## Usage
Run the main pipeline script from the project root directory:
python src/research_pipeline.py
The script will automatically fetch data from the World Bank API, perform feature engineering, estimate the models, and generate output files.

## Output
All results are saved in the output/ folder, organized by country (VN/, US/, CN/). Execution logs are stored in the logs/ directory.
