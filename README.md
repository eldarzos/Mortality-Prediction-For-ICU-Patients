# ICU Mortality Prediction using MIMIC-III

This project aims to predict in-hospital mortality for ICU patients using data from the MIMIC-III clinical database. The model leverages the first 48 hours of a patient's ICU stay, processing time-series data (vital signs, lab results, output events) through an LSTM-based neural network.

## Project Goal

To develop a dynamic, interpretable model that can predict the probability of in-hospital mortality for ICU patients on an hourly basis, utilizing the rich, raw event data available in Electronic Health Records (EHRs) without manual feature engineering. The goal is to provide a tool that could potentially aid in clinical decision-making and resource allocation.

## Features

* **Full EHR Utilization:** Processes a wide range of charted events, lab results, and output events.
* **Dynamic Risk Tracking:** The model is designed to output a mortality probability for each hour of the input window (first 48 hours).
* **Data-Driven Tokenization:** Continuous variables are binned using percentile-based quantization derived from the training data, and discrete variables are tokenized directly.
* **LSTM-based Model:** Utilizes Long Short-Term Memory networks to capture temporal patterns in patient data.

## Dataset

This project uses the [MIMIC-III (Medical Information Mart for Intensive Care III) v1.4 database](https://physionet.org/content/mimiciii/1.4/). Access to MIMIC-III is required and must be obtained through PhysioNet.

The input features for the model are derived from the first 48 hours of each patient's ICU stay.

## Pipeline Overview

The project follows a multi-step pipeline to process the raw MIMIC-III data and train the prediction model:

1.  **Preprocessing (Scripts `1_` to `8_`):**
    * `1_subject_events.py`: Extracts per-subject `stays.csv` (filtered ICU stay information) and `events.csv` (raw time-series events like CHARTEVENTS, LABEVENTS, OUTPUTEVENTS) from the MIMIC-III CSVs.
    * `2_validate_events.py`: Cleans `events.csv` by ensuring events can be linked to valid hospital admissions and ICU stays from `stays.csv`.
    * `3_subject2episode.py`: Segments data into individual episodes (stays). Creates `episode{i}.csv` (static data like LOS, Age, Mortality label) and `episode{i}_timeseries.csv` (events time-aligned to ICU admission).
    * `4_truncate_timeseries.py`: Truncates the `episode{i}_timeseries.csv` files to the first 48 hours of events and creates a combined `ITEMID_UOM` identifier. Output: `episode{i}_timeseries_48.csv`.
    * `5_split_train_test.py`: Splits the processed stays (based on paths to the 48-hour timeseries files) into training, validation, and test sets, stratified by the mortality label.
    * `6_generate_value_dict.py`: Analyzes the training set's 48-hour timeseries files to create a dictionary (`{t_hours}-{seed}-values.npy`) characterizing all observed `ITEMID_UOM` values (discrete vs. continuous distributions).
    * `7_quantize_events.py`: Uses the value dictionary to quantize continuous variables (into bins) and tokenize discrete variables. Overwrites the `episode{i}_timeseries_48.csv` files with these tokenized representations. Saves a token-to-index mapping (`{t_hours}_{seed}_{n_bins}-token2index.npy`).
    * `8_create_arrays.py`: Reads the tokenized timeseries files and the token map, converts tokens to integer indices, pads/truncates sequences to a fixed length, and saves the final `X` (input sequences), `Y` (mortality labels), and `paths` arrays into an `.npz` file for model training.

2.  **Model Training and Evaluation (`1_model_training_and_eval.ipynb`):**
    * Loads the preprocessed data arrays and token map.
    * Defines the LSTM-based model architecture (Embedder -> LSTM -> Decoder).
    * Includes a hyperparameter grid search loop.
    * Trains the model for each hyperparameter combination.
    * Evaluates on a validation set (using AUROC) for early stopping and model selection.
    * Evaluates the best model on the test set.
    * Logs results and saves trained models.

3.  **Token and Label Interpretation (`2_create_token_label_csv.ipynb`):**
    * This notebook likely loads the `token2index.npy` map and the MIMIC dictionary files (`D_ITEMS.csv`, `D_LABITEMS.csv`).
    * It generates a CSV file that maps each token string (and its integer index) to its clinical meaning (ITEMID label), its type (e.g., binned continuous, discrete), and its value range (if binned) or possible discrete values. This is crucial for interpreting model inputs and outputs.

4.  **Demo/Visualization (`3_create_demo.ipynb`):**
    * Loads a trained model and the processed test data.
    * Selects example patients (e.g., True Positive, False Positive, etc.).
    * Runs predictions for these patients and displays the sequence of events alongside the model's hourly mortality probability. This helps in understanding model behavior for specific cases.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Download MIMIC-III:** Obtain access to MIMIC-III v1.4 from PhysioNet and download the CSV files to a local directory (e.g., `/path/to/mimic-iii-clinical-database-1.4/`).
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure your `requirements.txt` file lists all necessary packages like `pandas`, `numpy`, `torch`, `scikit-learn`, `tqdm`).
4.  **Configure Paths:**
    * Update paths in the preprocessing scripts (e.g., `mimic3_path`, `output_path` in `1_subject_events.py` and subsequent scripts).
    * Update paths in the Jupyter notebooks (`PROJECT_ROOT`, `DATA_ROOT_DIR`, `RESULTS_DIR`, MIMIC dictionary paths).

## Running the Pipeline

1.  **Run Preprocessing Scripts:** Execute scripts `1_` through `8_` sequentially. Ensure the output directory of one script is the input for the next.
    ```bash
    python scripts/1_subject_events.py /path/to/mimic-iii-csvs /path/to/output_data_step1
    python scripts/2_validate_events.py /path/to/output_data_step1
    python scripts/3_subject2episode.py /path/to/output_data_step1 # Assuming script 2 overwrites in place
    python scripts/4_truncate_timeseries.py /path/to/output_data_step1 -t 48
    python scripts/5_split_train_test.py /path/to/output_data_step1 -t 48 -s 0
    python scripts/6_generate_value_dict.py /path/to/output_data_step1 -t 48 -s 0
    python scripts/7_quantize_events.py /path/to/output_data_step1 -t 48 -n 20 -s 0
    python scripts/8_create_arrays.py /path/to/output_data_step1 -t 48 -n 20 -s 0
    ```

2.  **Run Model Training:** Open and run the cells in `1_model_training_and_eval.ipynb`. Configure hyperparameters as needed.
3.  **Generate Token-Label Map:** Run `2_create_token_label_csv.ipynb` to create the interpretable token map.
4.  **Run Demo:** Run `3_create_demo.ipynb` to visualize predictions on example patients.

## Model Architecture

The core model consists of:

1.  **Embedding Layer:** Maps each token (representing a clinical event or a binned value) to a dense vector representation.
2.  **Temporal Aggregation:** Events within each hour are aggregated (e.g., averaged, potentially weighted) to form a single vector representing that hour.
3.  **LSTM Layer:** Processes the sequence of hourly vectors to capture temporal dependencies.
4.  **Decoder Layer:** A fully connected layer with a sigmoid activation function to output the probability of mortality.

## Results

The primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUROC). The project aims to achieve high AUROC on the test set, demonstrating good discrimination between patients who will survive and those who will not. Dynamic hourly predictions allow for tracking risk over time.

![MortalityLSTM_GridSearch_t48_lr0 0005_z64_h256_p0 1_wFalse_dTrue_seed0_hourly_auroc_with_baselines](https://github.com/user-attachments/assets/9e811452-95ac-4289-a386-5651239e9b92)



## Future Work / TODO

* Explore different RNN architectures (e.g., GRU, Transformers).
* Incorporate static baseline variables (e.g., age, gender at admission).
* Implement more sophisticated attention mechanisms.
* Extend prediction to other outcomes (e.g., Length of Stay, specific complications).
* Develop a more user-friendly interface for the demo.

## Contributing

(Add guidelines if you plan to have collaborators)

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**MIMIC-III Data Usage Notice**

This repository uses the MIMIC-III clinical database, which is not included here due to data usage restrictions.

To use this software, you must independently obtain access to the MIMIC-III v1.4 dataset via PhysioNet:  
https://physionet.org/content/mimiciii/1.4/

Use of the MIMIC-III data must comply with the **PhysioNet Credentialed Health Data License 1.5.0**, available here:  
https://physionet.org/content/mimiciii/1.4/LICENSE.txt

Do not redistribute any portion of the MIMIC-III dataset.

