from keras.models import load_model
import core
# Params
model_name = 'model'

save_file = str.format('resources/models/{}.h5', model_name)
model = load_model(core.save_file)

evaluation = model.evaluate(core.observations_x_test, core.observations_y_test, batch_size=core.batch_size, verbose=0)
print('Loss:', evaluation[0])
print('Metric values:', evaluation[1])
