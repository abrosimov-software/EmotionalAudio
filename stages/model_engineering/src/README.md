This service is responsible for all of the actiopns that are connected with model training, analysis and selection

#### API Endpoints

POST /updateDatasets/
Description: Accepts datasets for training models.
Parameters: 
datasets: List of datasets to be used for training.
Response: Indication message.

POST /train/
Description: Starts model training.
Parameters:  - 
Response: Indication message.

GET /candidates/
Description: Gathers information about the candidates.
Parameters:  - 
Response: List of candidates with performance metrics.

POST /validate/
Description: Analyzes the model performance.
Parameters:
candidate_name: Name of the model candidate to be validated.
Response: Indication message.

GET /getLeader/
Description: Returns the leader model.
Parameters: -
Response: Leader model