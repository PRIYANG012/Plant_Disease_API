from config import app, api, logger

from resources.Plants import Tomato_Prediction,Apple_Prediction,Cherry_Prediction,Corn_Prediction,Grape_Prediction,Peach_Prediction,Pepper_Prediction,Potato_Prediction,Strawberry_Prediction,Tomato_Prediction

"""
    Routing
"""

api.add_resource(Apple_Prediction, '/api/Apple_Prediction', endpoint='Apple_Prediction')
api.add_resource(Cherry_Prediction, '/api/Cherry_Prediction', endpoint='Cherry_Prediction')
api.add_resource(Corn_Prediction, '/api/Corn_Prediction', endpoint='Corn_Prediction')
api.add_resource(Grape_Prediction, '/api/Grape_Prediction', endpoint='Grape_Prediction')
api.add_resource(Peach_Prediction, '/api/Peach_Prediction', endpoint='Peach_Prediction')
api.add_resource(Pepper_Prediction, '/api/Pepper_Prediction', endpoint='Pepper_Prediction')
api.add_resource(Potato_Prediction, '/api/Cherry_Prediction', endpoint='Potato_Prediction')
api.add_resource(Strawberry_Prediction, '/api/Strawberry_Prediction', endpoint='Strawberry_Prediction')
api.add_resource(Tomato_Prediction, '/api/Tomato_Prediction', endpoint='Tomato_Prediction')





if __name__ == "__main__":
    app.run(debug=True)