# The Backtest App
I’ve built a simple cricket model for ratings batsmen and bowlers based on ball by ball test match data from cricsheet 
stretching back over the last 12 years. This file details some instructions for running the model within the jupyter 
notebook included in the repo

* Simple predictive cricket model using ball by ball test match data.ipynb

The model outputs ratings for batsmen and bowlers

# Requirements
Ensure the following packages are installed from the command line using pip3 if not already 
* pystan
* yaml
* tqdm
* requests 
* zipfile
* io

# Pulling files from repo
Pull the following files and save in the same directory locally
* cricket.stan
* cricket_data_fetcher.py
* cricket_model.py
* Predictive tennis model using point by point data from Grand Slams.ipynb

I've attached some pickled files containing a df of ball by ball data and metadata which can be fed into 
the model to save time fetching and parsing the data
* df_ball_by_ball.pkl
* df_metadata.pkl

The pickled files should also be saved in the same directory locally. You may or may not need to tweak your PYTHONPATH
to get cricket_data_fetcher.py and cricket_model.py to import smoothly within the notebook

# Running from within the notebook
The data can be fetched as follows
```
try:
    df_metadata = pd.read_pickle('df_metadata.pkl')
    df_ball_by_ball = pd.read_pickle('df_ball_by_ball.pkl')
except:
    zipped_files, list_of_yamls = cricket_data_fetcher.fetch_cricsheet_data()
    df_ball_by_ball, df_metadata = cricket_data_fetcher.extract_data_from_yamls(zipped_files, list_of_yamls)
```
The data will be loaded from the pickled files where present, otherwise it will be extracted more slowly from source

The model can be run as follows
```
df = CricketModel.fit(cricket_code=cricket_code,
                      df_ball_by_ball=df_ball_by_ball, 
                      df_metadata=df_metadata, 
                      fit_with_mcmc=False)
```
The batsmen and bowler ratings are encapsulated in the fit object and can be interrogated as follows

```
df.batsmen.query('games>=20').reset_index().head(15)
df.bowlers.query('games>=20').reset_index().head(15)
```
