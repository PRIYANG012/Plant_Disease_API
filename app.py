from config import app, api, logger

from resources.Plants import Plant_Prediction,Tomato_Prediction

"""
    Routing
"""

api.add_resource(Plant_Prediction, '/api/Prediction', endpoint='Plant_Prediction')
api.add_resource(Tomato_Prediction, '/api/Tomato_Prediction', endpoint='Tomato_Prediction')



if __name__ == "__main__":
    app.run(debug=True)