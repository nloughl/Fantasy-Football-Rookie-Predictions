# Fantasy Football Rookie Predictions

Predict how well an NFL rookie will perform in fantasy football (Half-PPR + TE Premium) based on their college stats and draft capital.

## Project Structure

```
Fantasy-Football-Rookie-Predictions/
├── 01_Data_gathering.ipynb    # Collect draft, college, and rookie NFL data
├── 02_Model.ipynb             # Train models and generate predictions
├── 03_Test.ipynb              # Prepare 2025 rookie class data for prediction
├── data/
│   ├── raw/                   # Draft CSVs from Pro Football Reference
│   ├── processed/             # Cleaned master datasets and college stats
│   └── output/                # Model predictions
├── .env                       # API keys (not tracked in git)
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   source .venv/bin/activate    # macOS/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your College Football Data API key:
   ```
   CFBD_API_KEY=your_key_here
   ```
   Get a free key at https://collegefootballdata.com/key

## Notebook Workflow

Run the notebooks in order:

1. **01_Data_gathering** - Loads NFL draft data (2019-2024), scrapes college stats from the CFBD API, fetches rookie season stats via `nfl_data_py`, calculates fantasy points, and builds `df_master.csv`.

2. **02_Model** - Trains four regression models (Random Forest, XGBoost, CatBoost, Ridge) on the master dataset and generates predictions for the 2025 rookie class.

3. **03_Test** - Prepares the 2025 draft class by pulling their college stats and physical measurements, producing `df_master_rookies_2025.csv` for use in notebook 02.

## Data Sources

- **NFL Draft data**: [Pro Football Reference](https://www.pro-football-reference.com/)
- **College football stats**: [College Football Data API](https://api.collegefootballdata.com/)
- **NFL rookie season stats**: [`nfl_data_py`](https://pypi.org/project/nfl-data-py/)

## Fantasy Scoring (Half-PPR + TE Premium)

| Category | Points |
|----------|--------|
| Passing TD | 4 |
| Passing Yards | 1 per 25 yds |
| Interception | -2 |
| Rushing/Receiving TD | 6 |
| Rushing Yards | 1 per 10 yds |
| Receiving Yards | 1 per 10 yds |
| Reception | 0.5 |
| TE Reception Bonus | +1 |
| Fumble Lost | -2 |
