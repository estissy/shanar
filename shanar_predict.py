import numpy as np
from keras.models import load_model

import core

model = load_model(core.save_file)

# Select last observations from data for prediction.
selected_observation_x = core.observations_x[-1]

predicted_values = []
for selected_forecast_time_lag in range(core.forecast_time_lags):
    reshaped_selected_observation_x = selected_observation_x.reshape(1, 1, -1)
    predicted = model.predict(reshaped_selected_observation_x, batch_size=core.batch_size, verbose=0)
    predicted_values += [predicted[0, 0]]
    selected_observation_x = np.hstack((selected_observation_x, predicted))
    selected_observation_x = selected_observation_x[:, 1:]

scaled_predicted_values = core.scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
scaled_predicted_values = scaled_predicted_values.flatten().tolist()
print(scaled_predicted_values)
