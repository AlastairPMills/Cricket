import pandas as pd
import numpy as np
import time
import tqdm
import yaml
import requests
from zipfile import ZipFile
from io import BytesIO


def logging(message):
    print(time.ctime() + ": " + message)


def fetch_cricsheet_data(zipfile_name='tests.zip'):
    """
    Function fetches specified zip file from https://cricsheet.org and returns a zip file object and list of yaml files
    :param zipfile_name: set to 'tests.zip' as default but can be switched to download 1-day, Twenty20 data etc.
    :return: zip file object and list of yaml files
    """
    r = requests.get(f'https://cricsheet.org/downloads/{zipfile_name}')
    zip_file_object = ZipFile(BytesIO(r.content))
    list_of_yamls = [file for file in zip_file_object.namelist() if 'yaml' in file]

    return zip_file_object, list_of_yamls


def parse_ball_by_ball_data_from_yaml(yaml_data):
    """
    Function parses and cleans yaml ball by ball data to pandas df format
    :param yaml_data: data from a single game in yaml format
    :return: df of ball by ball data
    """
    df = pd.DataFrame({})
    count = 0

    for idx1, item in enumerate(yaml_data['innings']):
        for idx2, ball in enumerate(item[list(item.keys())[0]]['deliveries']):
            for key in ball.keys():
                count = count + 1
                df.loc[count, 'innings'] = idx1 + 1
                df.loc[count, 'ball_count'] = key
                df.loc[count, 'batsman'] = ball[key]['batsman']
                df.loc[count, 'bowler'] = ball[key]['bowler']
                df.loc[count, 'non_striker'] = ball[key]['non_striker']
                df.loc[count, 'runs'] = ball[key]['runs']['batsman']
                df.loc[count, '4s'] = np.where(df.loc[count, 'runs'] == 4, 1, 0)
                df.loc[count, '6s'] = np.where(df.loc[count, 'runs'] == 6, 1, 0)
                df.loc[count, 'extras'] = ball[key]['runs']['extras']
                df.loc[count, 'runs_plus_extras'] = ball[key]['runs']['total']
                if 'wicket' in ball[key].keys():
                    df.loc[count, 'wicket'] = 1
                    df.loc[count, 'player_out'] = ball[key]['wicket']['player_out']
                    df.loc[count, 'wicket_kind'] = ball[key]['wicket']['kind']
                    if 'fielders' in ball[key]['wicket'].keys():
                        df.loc[count, 'fielder'] = ball[key]['wicket']['fielders'][0]
                    else:
                        df.loc[count, 'fielder'] = None
                else:
                    df.loc[count, 'wicket'] = 0
                    df.loc[count, 'player_out'] = None
                    df.loc[count, 'wicket_kind'] = None
                    df.loc[count, 'fielder'] = None

    return df


def parse_metadata_from_yaml(game_id, yaml_data):
    """
    Function parses and cleans yaml metadata for a specific game id
    :param game_id: cricsheet game id
    :param yaml_data: data from a single game in yaml format
    :return: metadata on specific game (e.g. home and away teams, toss winner etc.)
    """
    df_details = pd.DataFrame({})
    game_id = int(game_id)
    data = yaml_data.copy()

    try:
        df_details.loc[game_id, 'start_date'] = pd.to_datetime(data['info']['dates'][0])
        df_details.loc[game_id, 'home_team'] = data['info']['teams'][0]
        df_details.loc[game_id, 'away_team'] = data['info']['teams'][1]
        df_details.loc[game_id, 'match_type'] = data['info']['match_type']
        try:
            df_details.loc[game_id, 'winner'] = data['info']['outcome']['winner']
        except:
            df_details.loc[game_id, 'winner'] = data['info']['outcome']['result']
        try:
            df_details.loc[game_id, 'margin'] = data['info']['outcome']['by']['runs']
            df_details.loc[game_id, 'margin_type'] = 'runs'
        except:
            try:
                df_details.loc[game_id, 'margin'] = data['info']['outcome']['by']['wickets']
                df_details.loc[game_id, 'margin_type'] = 'wickets'
            except:
                df_details.loc[game_id, 'margin'] = None
                df_details.loc[game_id, 'margin_type'] = None
        df_details.loc[game_id, 'toss_decision'] = data['info']['toss']['decision']
        df_details.loc[game_id, 'toss_winner'] = data['info']['toss']['winner']
        df_details.loc[game_id, 'umpire_1'] = data['info']['umpires'][0]
        df_details.loc[game_id, 'umpire_2'] = data['info']['umpires'][1]
        df_details.loc[game_id, 'venue'] = data['info']['venue']
        df_details.loc[game_id, 'match_length'] = len(data['info']['dates'])
        df_details.loc[game_id, 'gender_type'] = data['info']['gender']
        try:
            df_details.loc[game_id, 'player_of_match'] = data['info']['player_of_match']
        except:
            df_details.loc[game_id, 'player_of_match'] = None
    except:
        logging(f"Metadata could not be extracted for {game_id}")

    return df_details


def extract_data_from_yamls(zip_file_object, list_of_yamls):
    """
    Function for looping through yaml files and parsing ball by ball and metadata
    :param zipped_files: zip file object extracted from https://cricsheet.org
    :param list_of_yamls: list of yaml files included in zip file object
    :return: df of ball by ball and metadata with game id keys
    """

    df_ball_by_ball = pd.DataFrame({})
    df_metadata = pd.DataFrame({})

    for i in tqdm.tqdm(range(len(list_of_yamls))):
        with zip_file_object.open(list_of_yamls[i]) as yamlfile:
            game_id = list_of_yamls[i].split('.')[0]
            yaml_data = yaml.load(yamlfile)
            try:
                df_ball_by_ball = pd.concat([df_ball_by_ball, parse_ball_by_ball_data_from_yaml(yaml_data)], axis=0)
                df_metadata = pd.concat([df_metadata, parse_metadata_from_yaml(game_id, yaml_data)], axis=0)
            except:
                logging(f"Problem parsing data for game {game_id}")

    return df_ball_by_ball, df_metadata
