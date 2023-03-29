import numpy as np
import pandas as pd
import plotly.express as px


def rebuild_timestamps(attention_df: pd.DataFrame, min_timestamp: int = None, max_timestamp: int = None, period_sec: float = 1.0) -> pd.DataFrame:
    '''For dataframe processing purposes: Creates new lines for missing timestamps in the dataframe with NA values
    --> min/max_timestamp: min and max timestamp to be considered - default=None: considers the min and max timestamps of the original data if not specified
    --> period_sec: theoretical period between 2 frame captures'''
    min = min_timestamp if min_timestamp is not None else attention_df['timestamp'].min()
    max = max_timestamp if max_timestamp is not None else attention_df['timestamp'].max()
    timestamps = range(min, max + 1, int(period_sec * 10))
    time_series = pd.Series(timestamps, name='timestamp')

    output_df = pd.merge(time_series, attention_df, how='left', on='timestamp')
    return output_df


def build_timegroups(attention_df: pd.DataFrame, grouping_factor: int = 10) -> pd.DataFrame:
    '''For dataframe processing purposes: Creates a time_group column that groups together XX successive timestamps
    --> grouping_factor: number of timestamps to be grouped together'''
    timestamps = sorted(attention_df["timestamp"].unique())
    timestamp_dict = {timestamps[i]: timestamps[i - i % grouping_factor] for i in range(len(timestamps))}
    time_group = attention_df['timestamp'].map(timestamp_dict)
    output_df = attention_df.drop(columns='time_group', errors='ignore')
    output_df.insert(2, 'time_group', time_group)
    return output_df


def process_audience_df(attention_df: pd.DataFrame, period_sec: float = 1.0, grouping_factor: int = 10):
    '''Generates final attention dataframe from the dataframe originally recorded:
    --> makes sure there is continuity in the data (no missing timestamp) - period_sec: theoretical period between 2 frame captures
    --> groups successive timestamps in one time_group bucket - grouping_factor: number of timestamps to be grouped together
    --> sorts the dataframe by timestamp'''
    output_df = rebuild_timestamps(attention_df, period_sec=period_sec)
    output_df = build_timegroups(output_df, grouping_factor=grouping_factor)
    output_df = output_df.sort_values(by=['timestamp', 'face_idx']).reset_index(drop=True)
    return output_df


def generate_person_df(attention_df: pd.DataFrame, person_name: str, period_sec: float = 1.0, grouping_factor: int = 10):
    '''Generates a person's attention dataframe from the dataframe originally recorded:
    --> filters the original dataframe on the specified person - person_name: name of the person as identified in the dataframe
    --> makes sure there is continuity in the data (no missing timestamp) - period_sec: theoretical period between 2 frame captures
    --> groups successive timestamps in one time_group bucket - grouping_factor: number of timestamps to be grouped together
    --> sorts the dataframe by timestamp'''

    person_mask = attention_df['recognition_prediction'] == person_name
    person_df = attention_df[person_mask]

    min_timestamp = attention_df['timestamp'].min()
    max_timestamp = attention_df['timestamp'].max()
    person_df = rebuild_timestamps(person_df, min_timestamp=min_timestamp, max_timestamp=max_timestamp, period_sec=period_sec)
    person_df = build_timegroups(person_df, grouping_factor=grouping_factor)
    person_df = person_df.sort_values(by=['timestamp']).reset_index(drop=True)

    return person_df


def average_series(s: pd.Series, na_limit: float = 0.4):
    '''For dataframe processing purposes: Takes a pandas series and returns the average value of the series if the share of non-NA values is higher to the na_limit'''
    na_rate = s.isna().sum() / len(s)
    return s.mean().round(2) if na_rate <= na_limit else np.nan


def plot_audience_attention(audience_df:pd.DataFrame, average_window: int = 1):
    '''Generates the audience's average attention curve (rolling average to smoothen the curve)
    -- roll: window of the rolling average - by default = 1 --> no rolling average'''
    df = audience_df.sort_values(by=['timestamp', 'face_idx']).reset_index(drop=True)
    original_ts_start = df['timestamp'][0]

    df_mean = df.groupby('timestamp')[['attentive']].mean()
    df_ma = df_mean.rolling(window=average_window).mean().dropna().reset_index()
    new_ts_start = df_ma['timestamp'][0]

    df_ma['timestamp'] = df_ma['timestamp'] - (new_ts_start - original_ts_start)                        # Reindexing the timestamps to start at zero
    df_ma["seconds"] = df_ma['timestamp'] / 10                                                          # Converts timestamp value to seconds
    df_ma["percentage"] = (df_ma['attentive'] * 100).astype(int)                                        # Converts [0;1] attention value to percentage

    max_sec = df_ma['seconds'].max()

    fig = px.line(df_ma, x="seconds", y="percentage", labels = dict(percentage = "Attentiveness (in %)", seconds = "Time (in seconds)"))
    fig.update_layout(yaxis_range=[0,100])
    fig.update_layout(xaxis_range=[0,max_sec])

    return fig.show()


def plot_individual_attention(person_df:pd.DataFrame, na_limit: float = 0.4):
    '''Generates a person's attention curve from the person's attention dataframe (average over the timegroup to smoothen the curve)
    -- na_limit: NA is returned instead of the average attention over the time_group if the NA values in the timegroup exceed the limit '''
    df_mean = person_df[['time_group', 'attentive']].groupby('time_group').agg(
        {'attentive': lambda s: average_series(s, na_limit)}).reset_index()

    df_mean["seconds"] = df_mean['time_group'] / 10
    df_mean["percentage"] = (df_mean['attentive'] * 100)

    max_sec = df_mean['seconds'].max()

    fig = px.line(df_mean, x="seconds", y="percentage", labels = dict(percentage = "Attentiveness (in %)", seconds = "Time (in seconds)"))
    fig.update_layout(yaxis_range=[0,100])
    fig.update_layout(xaxis_range=[0,max_sec])

    return fig.show()
