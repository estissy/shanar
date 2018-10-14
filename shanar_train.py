import keras as k

import core

model = k.Sequential()
model.add(k.layers.GRU(units=core.gru_units, input_shape=(1, core.number_of_time_lags)))
model.add(k.layers.Dense(1, activation=k.activations.linear))
model.compile(loss=k.losses.mean_squared_error, optimizer=core.optimizer,
              metrics=[k.losses.mean_absolute_error])

early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=core.early_stopping_patience,
                                           restore_best_weights=True)

model.fit(core.observations_x_train, core.observations_y_train, epochs=core.epochs, verbose=2,
          validation_data=(core.observations_x_test, core.observations_y_test), shuffle=False,
          callbacks=[early_stopping],
          batch_size=core.batch_size)

model.save(core.save_file)
