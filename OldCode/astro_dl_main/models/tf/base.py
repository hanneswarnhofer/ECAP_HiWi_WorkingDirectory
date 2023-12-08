from keras.models import Model


class BaseModel(Model):
    def get_exclude_keys(self, data_keys):
        if type(data_keys) == dict:
            data_keys = data_keys.keys()
        return [k for k in data_keys if k not in self.output_names]

    def get_exclude_label_keys(self, data_keys):
        if type(data_keys) == dict:
            data_keys = data_keys.keys()
        return [k for k in data_keys if k not in self.output_names + self.input_names]
