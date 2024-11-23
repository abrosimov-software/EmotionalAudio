# import os
# import io
# import yaml
# import torch
# import mlflow
# import mlflow.pytorch
# from flask import Flask, request, jsonify, send_file
# from torch import optim
# from src.utils import (
#     AnotherAudioDataset,
#     create_data_loaders,
#     train_model,
#     test_model,
# )
# from src.models.base_lstm_mpool import LSTMModel
# from src.models.adv_crnn import CRNN
# from src.models.adv_cnn_transf import CNNTransformerWithFeatures
# from src.models.adv_resnet import ResNetWithFeatures
# from src.models.adv_tcn import TCNWithFeatures

# # Load config
# with open("config/models.yaml", "r") as config_file:
#     try:
#         models_config = yaml.safe_load(config_file)
#     except yaml.YAMLError as exc:
#         print(exc)
#         models_config = {}

# # Initialize Flask app
# app = Flask(__name__)

# # MLflow experiment name
# experiment_name = "model_engineering_experiment"
# mlflow.set_experiment(experiment_name)

# # Get experiment ID
# experiment = mlflow.get_experiment_by_name(experiment_name)
# experiment_id = experiment.experiment_id

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data paths
# DATA_PATH = "/app/data/training_data"
# AUDIO_DIR = os.path.join(DATA_PATH, "audiofiles")
# FEATURES_DIR = os.path.join(DATA_PATH, "features")
# LABELS_DIR = os.path.join(DATA_PATH, "labels")

# # Existing endpoints...

# # Updated endpoint to get the best performing model's weights
# @app.route('/get_leader', methods=['GET'])
# def get_leader():
#     try:
#         # Retrieve all runs for the experiment
#         runs = mlflow.search_runs(experiment_ids=[experiment_id])

#         if runs.empty:
#             return jsonify({'error': 'No runs found in the experiment.'}), 404

#         # Assume best model is the one with the lowest test loss
#         # Sort runs by 'metrics.test_loss' in ascending order
#         runs = runs.sort_values('metrics.test_loss', ascending=True)

#         # Get the run ID of the best run
#         best_run = runs.iloc[0]
#         best_run_id = best_run['run_id']
#         best_model_type = best_run['params.model_type']

#         # Model URI
#         model_uri = f"runs:/{best_run_id}/models"

#         # Load the model using MLflow
#         model = mlflow.pytorch.load_model(model_uri)

#         # Serialize the model weights to a BytesIO object
#         buffer = io.BytesIO()
#         torch.save(model.state_dict(), buffer)
#         buffer.seek(0)

#         # Set the filename for the download
#         filename = f"{best_model_type}_best_model.pt"

#         # Send the file as a response
#         return send_file(
#             buffer,
#             as_attachment=True,
#             download_name=filename,
#             mimetype='application/octet-stream'
#         )

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # API endpoint to start a training run
# @app.route('/train', methods=['POST'])
# def train():
#     data = request.get_json()
#     model_type = data.get('model_type')
#     epochs = data.get('epochs', 10)
#     batch_size = data.get('batch_size', 32)

#     if model_type not in models_config:
#         return jsonify({'error': 'Model type not found in configuration.'}), 400

#     config = models_config[model_type]

#     # Start an MLflow run
#     with mlflow.start_run(run_name=model_type):
#         # Log parameters
#         mlflow.log_params({
#             'model_type': model_type,
#             'epochs': epochs,
#             'batch_size': batch_size,
#             **config.get('lstm', {}),
#             **config.get('cnn', {}),
#             **config.get('transformer', {}),
#             **config.get('tcn', {}),
#         })

#         # Prepare data loaders
#         dataset = AnotherAudioDataset(AUDIO_DIR)
#         train_loader, test_loader = create_data_loaders(dataset, batch_size=batch_size)

#         # Initialize model
#         if model_type == 'simple_lstm':
#             model = LSTMModel(config)
#         elif model_type == 'resnet':
#             model = ResNetWithFeatures(config)
#         elif model_type == 'crnn':
#             model = CRNN(config)
#         elif model_type == 'transformer':
#             model = CNNTransformerWithFeatures(config)
#         elif model_type == 'tcn_with_features':
#             model = TCNWithFeatures(config)
#         else:
#             return jsonify({'error': 'Unsupported model type.'}), 400

#         model = model.to(device)

#         # Optimizer
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         # Training loop
#         for epoch in range(epochs):
#             print(f'Epoch {epoch+1}/{epochs}')
#             train_loss = train_model(model, train_loader, optimizer, device)
#             test_loss = test_model(model, test_loader, device)

#             # Log metrics
#             mlflow.log_metric('train_loss', train_loss, step=epoch)
#             mlflow.log_metric('test_loss', test_loss, step=epoch)

#         # Log the model
#         mlflow.pytorch.log_model(model, 'models')

#         # Register the model
#         result = mlflow.register_model(
#             f'runs:/{mlflow.active_run().info.run_id}/models',
#             model_type,
#         )

#     return jsonify({'message': f'Training completed for model {model_type}.'}), 200

# # API endpoint to load the best model
# @app.route('/load_model', methods=['GET'])
# def load_model():
#     model_type = request.args.get('model_type')
#     stage = request.args.get('stage', 'Production')

#     try:
#         # Load model from MLflow Model Registry
#         model_uri = f'models:/{model_type}/{stage}'
#         model = mlflow.pytorch.load_model(model_uri)
#         print(f'Model {model_type} loaded from stage {stage}.')
#         return jsonify({'message': f'Model {model_type} loaded from stage {stage}.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # Run the app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

