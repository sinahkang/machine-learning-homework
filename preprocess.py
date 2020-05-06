import pandas as pd 

def prep_data(df):

    df = df.assign(HW=df["Height"] * df["Width"])
    df = df.assign(LL=df["Length3"]**2)
    df = df.assign(LHW=(df["LL"]) * (df["HW"]))
    
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop(["Length1", "Length2"], axis=1)
    
    feature_columns = [col for col in df.columns if col != "Weight"]

    X = df[feature_columns].values
    y = df["Weight"].values

    return X, y