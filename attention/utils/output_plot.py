import pandas as pd
import plotly.express as px
import nbformat

def reframe_dataframe(df:pd.DataFrame, video_start:int)-> pd.DataFrame:
    '''Reframes the dataframe to have axis in seconds and percentage'''
    df = df.reset_index()
    df["seconds"] = df['timestamp']/10 - video_start
    df["percentage"] = df['attentive']*100
    return df


def plot_attention_curve(df:pd.DataFrame, video_start:int):
    df = df.groupby('timestamp')[['attentive']].mean()
    df = df.rolling(window=10).mean().dropna()
    df = reframe_dataframe(df, video_start)
    fig = px.line(df, x="seconds", y="percentage", labels = dict(percentage = "Attentiveness (in %)", seconds = "Time (in seconds)"))
    fig.update_layout(yaxis_range=[0,100])
    fig.update_layout(xaxis_range=[0,150])
    return fig
