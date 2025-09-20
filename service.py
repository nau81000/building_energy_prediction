""" Service permettant de prédire les émissions totales de gaz à effet de serre (en tonnes métriques CO2e)

    Lancer `bentoml serve` pour utiliser le service
"""
import bentoml
import pandas
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import Enum

"""

Exemple de requête valide:

curl -k -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
    "input_data": {
        "NumberofFloors": 3,
        "LargestPropertyUseTypeGFA": 2500.0,
        "SecondLargestPropertyUseTypeGFA": 1000.0,
        "ENERGYSTARScore": 50.0, 
        "SteamUsed": true,
        "NaturalGasUsed": true,
        "PrimaryPropertyType": "Hotel",   
        "Neighborhood": "Central"
    }
}'

Exemple de requête non valide:

curl -k -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
    "input_data": {
        "NumberofFloors": -1,
        "LargestPropertyUseTypeGFA": 0.0,
        "SecondLargestPropertyUseTypeGFA": 0.0,
        "ENERGYSTARScore": 0.0, 
        "SteamUsed": -1,
        "NaturalGasUsed": -1,
        "PrimaryPropertyType": "",   
        "Neighborhood": ""
    }
}'
"""

mappings = { 
    'PrimaryPropertyType': {
        'distribution center': 0, 'hospital': 1, 'hotel': 2, 'k-12 school': 3,
        'large office': 4, 'low-rise multifamily': 5, 'medical office': 6, 'mixed use property': 7,
        'other': 8, 'refrigerated warehouse': 9, 'residence hall': 10, 'retail store': 11,
        'senior care community': 12, 'small- and mid-sized office': 13, 'supermarket / grocery store': 14,
        'warehouse': 15, 'worship facility': 16
    },
    'Neighborhood': {
        'ballard': 0, 'central': 1, 'delridge': 2, 'downtown': 3, 'east': 4,
        'greater duwamish': 5, 'lake union': 6, 'magnolia / queen anne': 7, 'north': 8, 'northeast': 9,
        'northwest': 10, 'southeast': 11, 'southwest': 12
    },
}

# Colonnes
model_columns = [
    'NumberofFloors',
    'LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA',
    'ENERGYSTARScore',
    'SteamUsed', 'NaturalGasUsed',
    'PrimaryPropertyType_encoded',
    'Neighborhood_encoded'
]

# Définition du format JSON attendu
class InputData(BaseModel):
    NumberofFloors: int = Field(..., ge=0, description="Le nombre d'étages du/des bâtiment(s).")
    LargestPropertyUseTypeGFA: float = Field(..., gt=0, description="La surface construite principal du/des bâtiment(s) (pieds carrés).")
    SecondLargestPropertyUseTypeGFA: float = Field(..., gt=0, description="La surface construite secondaire du/des bâtiment(s) (pieds carrés).")
    ENERGYSTARScore: float = Field(..., ge=1, le=100, description="Le score ENERGY STAR (entre 1 et 100).")
    SteamUsed: bool = Field(..., description="La source d'énergie vapeur est-elle utilisée ou non.")
    NaturalGasUsed: bool = Field(..., description="La source d'énergie gaz naturel est-elle utilisée ou non.")
    PrimaryPropertyType: str
    Neighborhood: str

    @field_validator('PrimaryPropertyType')
    def check_primarypropertytype(cls, v):
        if v.lower() not in mappings['PrimaryPropertyType']:
            raise ValueError(f"PrimaryPropertyType '{v}' is not allowed. Allowed values are:{list(mappings['PrimaryPropertyType'].keys())}")
        return v

    @field_validator('Neighborhood')
    def check_neighborhood(cls, v):
        if v.lower() not in mappings['Neighborhood']:
            raise ValueError(f"Neighborhood '{v}' is not allowed. Allowed values are:{list(mappings['Neighborhood'].keys())}")
        return v

@bentoml.service(http={"port": 8080})
class Prediction:
    def __init__(self) -> None:
        # Charger le modèle enregistré
        self.pipeline = bentoml.sklearn.load_model("rf_pipeline_model:latest")

    @bentoml.api
    def predict(self, input_data: dict):
        # Valider les entrées
        try:
            data = InputData(**input_data)
        except ValidationError as e:
            # Reformater les erreurs de manière plus lisible
            errors = [
                {"field": ".".join(str(loc) for loc in err["loc"]), "error": err["msg"]}
                for err in e.errors()
            ]
            return {"error": "Invalid input", "details": errors}

        features = [
            data.NumberofFloors,
            data.LargestPropertyUseTypeGFA,
            data.SecondLargestPropertyUseTypeGFA,
            data.ENERGYSTARScore,
            data.SteamUsed,
            data.NaturalGasUsed,
            mappings['PrimaryPropertyType'][data.PrimaryPropertyType.lower()],
            mappings['Neighborhood'][data.Neighborhood.lower()]
        ]
        df = pandas.DataFrame([features])
        df.columns = model_columns
        prediction = self.pipeline.predict(df)[0]
        return {"prediction (kBtu)": prediction}
