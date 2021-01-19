import pandas as pd
import os

def save_or_append(df_output, path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df = df.append(df_output, ignore_index = True)
    else :
        df = df_output
    df.to_csv(path, index = False)
