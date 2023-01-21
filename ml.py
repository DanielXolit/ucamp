
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np
import pickle
#from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

api = Api(app,
          title='Generación predicción',
          descripcion='Api para la predicción de supervivencia en el titanic')

ns = api.namespace('predict',description='Generación de predicción')

parser = api.parser()

parser.add_argument(
    'age',
    type=int,
    required=True,
    help='Años de la persona',
    location='args')

parser.add_argument(
    'fare',
    type=float,
    required=True,
    help='Precio del tiquete pagado',
    location='args')

parser.add_argument(
    'pclass',
    choices=[1,2,3],
    type=int,
    required=True,
    help='Pclass',
    location='args')

parser.add_argument(
    'sex_female',
    choices=[0,1],
    type=int,
    required=True,
    help='sex_female',
    location='args')

parser.add_argument(
    'sex_male',
    choices=[0,1],
    type=int,
    required=True,
    help='sex_male',
    location='args')

resource_fields = api.model('Resource', {'respuesta': fields.String,})

def pred(age,fare,pclass,sex_female,sex_male):
    
    pclass_1 = 0
    pclass_2 = 0
    pclass_3 = 0
    if pclass == 1:
        pclass_1 = 1
    elif pclass == 2:
        pclass_2 = 1
    else:
        pclass_3 = 1

    arreglo = np.array([age,fare,pclass_1,pclass_2,pclass_3,sex_female,sex_male])
    classifier = pickle.load(open(r"C:\Users\leon-\Downloads\venv\classifier.pkl", 'rb'))
    pred = classifier.predict(arreglo.reshape(1,-1))[0]   
    if pred == 1:
        respuesta = "TRUE: El pasajero pudo haber sobrevivido"
    else:
        respuesta = "FALSE: El pasajero pudo NO haber sobrevivido"
    return respuesta

@ns.route('/')
class DSApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    
    def get(self):
        args = parser.parse_args()
        return {'respuesta':pred(args.age,args.fare,args.pclass,args.sex_female,args.sex_male)},200

if __name__ == "__main__":
     app.run(debug=True,use_reloader=False,port=5000)
