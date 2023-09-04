# Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN) with Graph Attention Networks

In this repository, I present my implementation of the Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN) model, with a focus on time series prediction (source paper: https://arxiv.org/abs/1704.02971). This personal project allowed me to delve into the intricacies of this model and explore its potential by incorporating Graph Attention Networks (GAT) and handling multiple features per timestep.

## Key Features
- `Graph Attention Networks (GAT)`: This implementation extends the DA-RNN model by incorporating GAT into the architecture. This addition enables the model to capture complex relationships between different time series data, possibly enhancing its predictive capabilities.

- `Multiple Features per Timestep`: Unlike the original DA-RNN, this version accommodates multiple features per timestep, allowing for a more comprehensive analysis of the input data.

## Project Structure

The project is organized into several Python files:

- `dataset.py`: Contains functions for preparing the dataset.
- `model.py`: Defines the DA-RNN model architecture and initialization.
- `train.py`: Contains functions for training the model, including the training loop, validation, and early stopping.
- `main.py`: The main script that uses all the above modules to perform the actual training.

In addition, you will find an accompanying Jupyter Notebook file, `DAARN-GAT.ipynb`, which serves as a convenient resource for recording and revisiting key project checkpoints and insights.

## Usage

To run this project, ensure that you have all the necessary dependencies installed. Additionally, please make sure you have the following prerequisites:

`financetoolkit`: You will need the financetoolkit library. 

`FinancialModelingPrep API Key`: To access financial data, you will require an API key from FinancialModelingPrep.

## Acknowledgements

I would like to thank KurochkinAlexey for his excellent implementation of the DA-RNN model, which served as a great inspiration for this project.
