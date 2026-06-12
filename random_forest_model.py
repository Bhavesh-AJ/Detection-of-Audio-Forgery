from sklearn.ensemble import RandomForestClassifier


def build_random_forest():

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    return model