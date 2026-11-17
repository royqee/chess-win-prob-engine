import pandas as pd
import numpy as np

class ChessDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load_and_clean(self):
        self.df = pd.read_csv(self.file_path)
        self.df.dropna(subset=['winner', 'white_rating', 'black_rating', 'opening_eco', 'moves'], inplace=True)
        self.df.drop_duplicates(subset=['id'], inplace=True)
        self.df = self.df[self.df['winner'] != 'draw'].copy()
        self.df['white_win'] = self.df['winner'].map({'white': 1, 'black': 0})
        self.df['white_win'] = self.df['white_win'].astype(int)
        return self.df

class FeatureExtractor:
    def __init__(self):
        self.features_cols = ['elo_diff', 'avg_rating', 'white_castled_10', 'black_castled_10', 'white_early_queen_move', 'black_early_queen_move', 'eco_group']
        self.eco_groups = ['A', 'B', 'C', 'D', 'E']
        
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        df= df.copy()
        df['elo_diff'] = df['white_rating'] - df['black_rating']
        df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2
        df['white_moves'] = df.apply(lambda row: row['moves'].split()[::2], axis=1)
        df['black_moves'] = df.apply(lambda row: row['moves'].split()[1::2], axis=1)
        df['white_castled_10'] = df['white_moves'].apply(lambda moves: any(move.startswith('O-O') for move in moves[:10])).astype(int)
        df['black_castled_10'] = df['black_moves'].apply(lambda moves: any(move.startswith('O-O') for move in moves[:10])).astype(int)
        df['white_early_queen_move'] = df['white_moves'].apply(lambda moves: any(move.startswith('Q') for move in moves[:10])).astype(int)
        df['black_early_queen_move'] = df['black_moves'].apply(lambda moves: any(move.startswith('Q') for move in moves[:10])).astype(int)
        df['eco_group'] = pd.Categorical(df['opening_eco'].str[0], categories=self.eco_groups)
        return df[self.features_cols]



