import pandas as pd
import plotly.express as px
import nbformat

def plot_attention_curve(attention_df:pd.DataFrame, roll: int = 10):
    df = attention_df.sort_values(by=['timestamp', 'face_idx']).reset_index(drop=True)

    df_mean = df.groupby('timestamp')[['attentive']].mean()
    df_ma = df_mean.rolling(window=roll).mean().dropna().reset_index()

    min_timestamp = df_ma['timestamp'].min() / 10
    df_ma["seconds"] = df_ma['timestamp'] / 10 - min_timestamp
    df_ma["percentage"] = (df_ma['attentive'] * 100).astype(int)
    max_sec = df_ma['seconds'].max()

    fig = px.line(df_ma, x="seconds", y="percentage", labels = dict(percentage = "Attentiveness (in %)", seconds = "Time (in seconds)"))
    fig.update_layout(yaxis_range=[0,100])
    fig.update_layout(xaxis_range=[0,max_sec])

    return fig
