from sklearn.model_selection import train_test_split
from features import ChessDataLoader, FeatureExtractor
from estimator import WinProbEstimator


def main():
    #1. load and clean the chess games dataset
    loader = ChessDataLoader("games.csv")
    df = loader.load_and_clean()

    # 2. Extract features and target variable
    extractor = FeatureExtractor()
    X = extractor.transform(df)
    y = df["white_win"]

    # 3. train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. train the model and evaluate
    estimator = WinProbEstimator()
    estimator.fit(X_train, y_train)
    estimator.evaluate(X_test, y_test)

    # 5. predict the probability of white winning for two different scenarios
    prob_solid = estimator.predict_match(
        white_elo=1500,
        black_elo=1500,
        opening_moves="e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6",
        opening_eco="C50",
    )
    prob_blunder = estimator.predict_match(
        white_elo=1500,
        black_elo=1500,
        opening_moves="e4 e5 Qh5 Nc6 Bc4 g6 Qf3",
        opening_eco="C20",
    )

    print(f"Equal Elo + Early Castling White Win Prob: {prob_solid:.2%}")
    print(f"Equal Elo + Early Queen Move White Win Prob: {prob_blunder:.2%}")


if __name__ == "__main__":
    main()