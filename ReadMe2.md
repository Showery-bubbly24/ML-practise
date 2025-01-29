# Dashboard

#### Example of dashboard app
```python
from sqlalchemy import create_engine
import pandas as pd
import dash
from dash import dcc, html as h, Input, Output, dash_table
import plotly.express as px

engine = create_engine('sqlite:///Database',echo=False)
df = pd.read_sql('Select * from data', con=engine)
df['date'] = pd.to_datetime(df['date'])
df_dop = pd.read_csv('path_to_file')
df_dop = df_dop[['station_id', 'station_name']]
df = pd.merge(df, df_dop, on='station_id', how='left')
df['station_name'] = df['station_name'].fillna('Another')


app = dash.Dash(__name__)
app.layout = h.Div([
    h.H1('Analytics dashboard'),
    h.Div([
        dcc.DatePickerRange(
            id='date-range',
            start_date=df['date'].min().date(),
            end_date=df['date'].max().date(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Dropdown(
            id='station_dropdown',
            options=[{'label': station, 'value': station} for station in df['station_name'].unique()],
            placeholder='Choose station',
            multi=True
        )
    ]),

    h.Div([
        h.H3('Graph of station load'),
        dcc.Graph(id='station-load-graph')
    ]),

    h.Div([
        h.H3('Table with mean load'),
        dash_table.DataTable(
            id='real-bandwidth-table',
            columns=[
                {'name': 'Station Name', 'id': 'station_name'},
                {'name': 'Mean value of peoples', 'id': 'num_val'}
            ]
        )
    ]),

    h.Div([
        h.H3('Top of tables'),
        dash_table.DataTable(
            id='top-stations-table',
            columns=[
                {'name': 'Station Name', 'id': 'station_name'},
                {'name': 'Number of peoples', 'id': 'num_val'}
            ]
        )
    ]),

    h.Div([
        h.H3(id='mean-stats-summary', style={'textAlign': 'center'})
    ])
])

@app.callback(
    [Output('station-load-graph', 'figure'),
        Output('real-bandwidth-table', 'data'),
        Output('top-stations-table', 'data'),
        Output('mean-stats-summary', 'children')],
    [Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('station_dropdown', 'value')],
)

def update_dashboard(start_date, end_date, selected_stations):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if selected_stations:
        filtered_df = filtered_df[filtered_df['station_name'].isin(selected_stations)]

    # 1) Load
    station_max = filtered_df.groupby('station_id')['num_val'].max().rename('max_num_val')
    filtered_df = filtered_df.merge(station_max, on='station_id')
    filtered_df['load_percentage'] = (filtered_df['num_val'] / filtered_df['max_num_val']) * 100
    load_fig = px.line(
        filtered_df, x='date', y='load_percentage', color='station_name',
        labels={'date': 'Date', 'load_percentage': 'Load lvl'}
    )

    # 2) Real bandwidth
    dndw  = filtered_df.groupby('station_name')['num_val'].mean().reset_index()

    # 3) Top of station
    top_df = filtered_df.groupby('station_name')['num_val'].sum().nlargest(5).reset_index()

    # 4) Dop info
    avg = filtered_df['num_val'].mean()
    summary = f'Mean value: {avg:.2f}'

    return load_fig, dndw.to_dict('records'), top_df.to_dict('records'), summary


if __name__ == '__main__':
    app.run_server(debug=True)
```


# Clusterization

#### My cluster
```python
class FavoClusterization:
    def __init__(self, data, start_date, end_date):
        self.filtered_df = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()

    def fit_predict(self):
        quantiles = self.filtered_df['num_val'].quantile([0.33, 0.66, 1])
        self.filtered_df['group_id'] = pd.cut(
            self.filtered_df['num_val'],
            bins=[-float('inf'),
                  quantiles.loc[0.33],
                  quantiles.loc[0.66],
                 float('inf')],
            labels=[0, 1, 2]
        )

        self.filtered_df = self.filtered_df.drop('index', axis=1)
        return self.filtered_df
```

#### Visualization
```python
plt.figure(figsize=(12, 6))
plt.subplot(122)
plt.scatter(new_data['hour'], new_data['num_val'], c=new_data['group_id'], cmap='viridis')
plt.title("Кластеризация")
plt legend()
plt.show()
```

#### Mean and median values
```python
print('Means')
print(f'Group 0: {new_data[new_data['group_id'] == 0]['num_val'].mean()}',
    f'\nGroup 1: {new_data[new_data['group_id'] == 1]['num_val'].mean()}',
    f'\nGroup 2: {new_data[new_data['group_id'] == 2]['num_val'].mean()}'
)

print('Medians')
print(f'Group 0: {new_data[new_data['group_id'] == 0]['num_val'].median()}',
    f'\nGroup 1: {new_data[new_data['group_id'] == 1]['num_val'].median()}',
    f'\nGroup 2: {new_data[new_data['group_id'] == 2]['num_val'].median()}'
)
```
