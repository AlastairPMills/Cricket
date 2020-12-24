import pandas as pd
import numpy as np
from collections import namedtuple
import datetime
import pystan
import cricket_data_fetcher


class CricketModel:
    def __init__(self, summary, debug_tables, skill_ratings):
        self.summary = summary
        self.debug_tables = debug_tables
        self.skill_ratings = skill_ratings

    TEST_TEAMS = ['Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'Ireland', 'New Zealand',
                  'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe']
    DATE_CUTOFF = '2008-01-01'

    SIGMA_BAT = 1.5
    SIGMA_BOWL = 1.5
    STARTING_GUESS_BAT = 0
    STARTING_GUESS_BOWL = 0
    DISCOUNT_COEFFICIENT = 1

    RUN_WEIGHTS = {
        0: 0.25,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7
    }

    ITERATIONS = 500
    CHAINS = 4

    CricketModelOutput = namedtuple("CricketModelOutput", ['games', 'ball_by_ball_df', 'batsmen',
                                                           'bowlers', 'stan_input', 'summary'])

    @staticmethod
    def prepare_data(df_ball_by_ball, df_metadata):

        df_clean = (
            df_ball_by_ball.set_index('game_id')
            .join(df_metadata.set_index('game_id'), how='left')
            .reset_index()
            .rename(columns={'index': 'game_id'})
            .assign(batting_team=lambda df: df.pipe(CricketModel.extract_batting_team))
            .assign(bowling_team=lambda df: (
                np.where(df['batting_team'] == df['home_team'], df['away_team'], df['home_team'])))
            .assign(batsman_id_stan=lambda df: df['batsman'].factorize(sort=False)[0] + 1)
            .assign(bowler_id_stan=lambda df: df['bowler'].factorize(sort=False)[0] + 1)
            .assign(result=lambda df: df['wicket'].astype(int))
            .assign(time_weight=lambda df: (
                    CricketModel.DISCOUNT_COEFFICIENT ** (datetime.datetime.now() - df['start_date']).dt.days))
            .assign(final_weight=lambda df: df['runs'].map(CricketModel.RUN_WEIGHTS) * df['time_weight'])
        )

        df_clean['starting_guess_bat'] = CricketModel.STARTING_GUESS_BAT
        df_clean['starting_guess_bowl'] = CricketModel.STARTING_GUESS_BOWL
        df_clean['sigma_bat'] = CricketModel.SIGMA_BAT
        df_clean['sigma_bowl'] = CricketModel.SIGMA_BOWL

        batsmen, bowlers = CricketModel.extract_bowlers_and_batters(df_clean)

        stan_input = {
            'N': len(df_clean),
            'B': len(batsmen),
            'K': len(bowlers),
            'batsman': df_clean['batsman_id_stan'],
            'bowler': df_clean['bowler_id_stan'],
            'result': df_clean['result'],
            'weight': df_clean['final_weight'],
            'starting_guess_bat': batsmen['starting_guess'],
            'starting_guess_bowl': bowlers['starting_guess'],
            'sigma_bat': 2,
            'sigma_bowl': 2
        }

        return df_clean, batsmen, bowlers, stan_input

    @staticmethod
    def fit(cricket_code, df_ball_by_ball=None, df_metadata=None, fit_with_mcmc=False):

        if (df_ball_by_ball is None) | (df_metadata is None):
            zipped_files, list_of_yamls = cricket_data_fetcher.fetch_cricsheet_data()
            df_ball_by_ball, df_metadata = cricket_data_fetcher.extract_data_from_yamls(zipped_files, list_of_yamls)

        df_clean, batsmen, bowlers, stan_input = CricketModel.prepare_data(df_ball_by_ball, df_metadata)

        stan_model = pystan.StanModel(model_code=cricket_code, verbose=False)

        if fit_with_mcmc:
            fit_object = stan_model.sampling(data=stan_input, iter=CricketModel.ITERATIONS, chains=CricketModel.CHAINS)
        else:
            fit_object = stan_model.optimizing(data=stan_input)

        stan_fit_object = StanFit(fit_object, was_fit_with_mcmc=fit_with_mcmc)

        batsman_nations, bowler_nations = CricketModel.extract_nationalities(df_clean)
        common_cols = ['team', 'games', 'ba', 'sr', 'rating', 'rating_std']

        batsmen = (
            batsmen
            .assign(team=lambda df: df.batsman.map(batsman_nations))
            .assign(rating=lambda df: stan_fit_object.get_parameter_mean('bat_ability'))
            .assign(rating_std=lambda df: stan_fit_object.get_parameter_std('bat_ability'))
            .rename(columns={'wickets': 'outs'})
            .sort_values(by='rating', ascending=True)
            [['batsman'] + common_cols]
        )

        bowlers = (
            bowlers
            .assign(team=lambda df: df.bowler.map(bowler_nations))
            .assign(rating=lambda df: stan_fit_object.get_parameter_mean('bowl_ability'))
            .assign(rating_std=lambda df: stan_fit_object.get_parameter_std('bowl_ability'))
            .sort_values(by='rating', ascending=True)
            [['bowler'] + common_cols]
        )

        batsmen_rating_dict = dict(zip(batsmen.index, batsmen.rating))
        bowlers_rating_dict = dict(zip(bowlers.index, bowlers.rating))

        summary = stan_fit_object.get_summary()

        df_clean = (
            df_clean
            .assign(base_rate=stan_fit_object.get_parameter_mean('base_rate'))
            .assign(batsman_rating=lambda df: df['batsman_id_stan'].map(batsmen_rating_dict))
            .assign(bowler_rating=lambda df: df['bowler_id_stan'].map(bowlers_rating_dict))
            .assign(wicket_prob=lambda df: (
                    1 / (1 + np.exp(-(df['base_rate'] + df['batsman_rating'] - df['bowler_rating'])))))
        )

        return CricketModel.CricketModelOutput(games=df_metadata,
                                               ball_by_ball_df=df_clean,
                                               batsmen=batsmen,
                                               bowlers=bowlers,
                                               stan_input=stan_input,
                                               summary=summary)

    @staticmethod
    def extract_bowlers_and_batters(df_master):

        batsmen_groupby, bowlers_groupby = (
            df_master.groupby(pos_type)
            .agg({
                'game_id': 'nunique',
                'runs': 'sum',
                'ball_count': 'count',
                'wicket': 'sum'})
            for pos_type in ['batsman', 'bowler']
        )

        batsmen, bowlers = (
            pd.Series(df_master[pos_type].factorize(sort=False)[0] + 1, index=df_master[pos_type])
            .reset_index()
            .drop_duplicates()
            .rename(columns={0: f'{pos_type}_id'})
            .set_index(f'{pos_type}')
            .assign(starting_guess=pos_guess)
            .join(pos_gpby)
            .assign(avg_runs_per_wicket=lambda df: np.round(df['runs'] / df['wicket'], 2))
            .assign(avg_balls_per_wicket=lambda df: np.round(df['ball_count'] / df['wicket'], 2))
            .rename(columns={
                'game_id': 'games',
                'wicket': 'wickets',
                'avg_runs_per_wicket': 'ba',
                'avg_balls_per_wicket': 'sr'})
            .reset_index()
            .set_index(f'{pos_type}_id')
            .assign(wickets=lambda df: df['wickets'].astype(int))
            .drop(columns=['ball_count', 'runs'])
            for pos_type, pos_gpby, pos_guess in zip(['batsman', 'bowler'],
                                                     [batsmen_groupby, bowlers_groupby],
                                                     [CricketModel.STARTING_GUESS_BAT,
                                                      CricketModel.STARTING_GUESS_BOWL])
        )

        return batsmen, bowlers

    @staticmethod
    def extract_nationalities(df_master):

        batsman_teams = dict(zip(df_master.loc[lambda df: df['innings'].isin([1, 2])].batsman,
                                 df_master.loc[lambda df: df['innings'].isin([1, 2])].batting_team))

        bowler_teams = dict(zip(df_master.loc[lambda df: df['innings'].isin([1, 2])].bowler,
                                df_master.loc[lambda df: df['innings'].isin([1, 2])].bowling_team))

        return batsman_teams, bowler_teams

    @staticmethod
    def extract_batting_team(df):

        df['toss_loser'] = np.where(df['toss_winner'] == df['home_team'], df['away_team'], df['home_team'])
        df['inn_1_team_batting'] = np.where((df['toss_decision'] == 'bat'), df['toss_winner'], df['toss_loser'])
        df['inn_2_team_batting'] = np.where((df['toss_decision'] == 'bat'), df['toss_loser'], df['toss_winner'])
        df['batting_team'] = np.where(df['innings'] == 1, df['inn_1_team_batting'], (
            np.where(df['innings'] == 2, df['inn_2_team_batting'], None)))

        return df['batting_team']


class StanFit:
    """
    This class exists to provide a standard interface for extracting point estimates of parameters from STAN models
    regardless of whether they were fitted with optimisation or sampling methods.
    """
    def __init__(self, fit, was_fit_with_mcmc):
        self.fit = fit
        self.was_fit_with_mcmc = was_fit_with_mcmc

    def get_parameter_mean(self, parameter_name):
        if self.was_fit_with_mcmc is True:
            return self.fit[parameter_name].mean(axis=0)
        else:
            return self.fit[parameter_name]

    def get_parameter_std(self, parameter_name):
        if self.was_fit_with_mcmc is True:
            return self.fit[parameter_name].std(axis=0)
        else:
            return None

    def get_summary(self):
        if self.was_fit_with_mcmc is True:
            summary = self.fit.stansummary(pars=['lp__'])
            return summary
        else:
            return None
