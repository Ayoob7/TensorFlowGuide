import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt
#plt.style.use('')

df = pd.read_csv('monthly.csv')

df['Month'] = pd.DatetimeIndex(df['Month'])
df = df.rename(columns={'Month': 'ds','Monthly': 'y'})
print(df.head(3))
print(df.dtypes)

ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly mean thickness (Dobson units) ozone column Arosa, Switzerland 1926-1971')
ax.set_xlabel('Date')



my_model = Prophet(interval_width=0.95)

my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=36, freq='MS')

forecast = my_model.predict(future_dates)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
my_model.plot(forecast,uncertainty=True)
my_model.plot_components(forecast)
plt.show()