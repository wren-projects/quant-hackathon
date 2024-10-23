from analysis import Data
import predict
import bet

data_path = ""

data = Data(data_path)
model = predict.train_model(data)