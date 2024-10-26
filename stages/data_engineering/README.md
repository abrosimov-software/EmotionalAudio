This service is responsible for all of the actions that are connected with data preprocessing and preparation.

#### API Endpoints
POST /extract_data/
Description: Extracts data from a specified source.
Parameters:
source: (Optional) The data source (path to data folder). If not provided, it uses the default source.
Response: Indication message

POST /process_data/
Description: Processes data that was either extracted using extract_data or provided by the user.
Parameters:
user_data: (Optional) Data to be processed. If not provided, the service processes the data previously extracted.
Response: Indication message + processed data.

GET /serve/
Description: Serves the processed dataset for training the model.
Parameters: -
Response: Indication message + processed dataset.