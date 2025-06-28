# ICU Mortality Prediction using MIMIC-III

This project predicts **in-hospital mortality** for ICU patients using the MIMIC-III clinical database.  
The model ingests the **first 48 hours** of each ICU stay—vital signs, labs, and outputs—and feeds them to an **LSTM-based neural network** that yields an **hourly risk trajectory**.

---

## Project Goal

Develop a dynamic, interpretable model that updates a patient's mortality probability **hour-by-hour**, using raw EHR event streams and **no manual feature selection**, so clinicians can make earlier, data-driven decisions.

---

## Features

* **Full EHR Utilisation** – charted events, labs, outputs, everything.
* **Dynamic Risk Tracking** – probability every hour for the first 48 h.
* **Data-Driven Tokenisation** – continuous variables binned by percentiles; discrete ones tokenised directly.
* **LSTM Backbone** – captures temporal dependencies without handcrafted features.

---

## Dataset

* **Source:** [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/)  
* **Window:** first 48 h of ICU stay (configurable)

> **Access note:** You must request MIMIC-III access via PhysioNet and accept the **PhysioNet Credentialed Health Data License**. The raw data are **not** included in this repo.

---

## Token Creation and Embedding Process 🧬

This section explains how raw clinical events are converted to tokens and then embedded for neural network training.

### 1. Token Creation Examples

Clinical events are converted to tokens using the `ITEMID_UOM` (clinical variable + unit of measurement) as the key:

#### **Continuous Variables (≥20 unique values)**
Raw clinical data is binned into percentiles:
```
Heart Rate (ITEMID: 220045, UOM: bpm)
- Raw values: [65, 78, 92, 110, 95, ...]
- Percentile bins (n_bins=20): [0-5%, 5-10%, ..., 95-100%]
- Tokens: "220045_bpm:0", "220045_bpm:7", "220045_bpm:15", "220045_bpm:19", ...

Blood Pressure Systolic (ITEMID: 220050, UOM: mmHg)
- Raw values: [120, 135, 95, 160, ...]
- Tokens: "220050_mmHg:8", "220050_mmHg:12", "220050_mmHg:2", "220050_mmHg:19", ...
```

#### **Discrete Variables (<20 unique values)**
Values are tokenized directly:
```
Ventilator Mode (ITEMID: 720, UOM: "")
- Raw values: ["SIMV", "CPAP", "PSV", ...]
- Tokens: "720_:SIMV", "720_:CPAP", "720_:PSV", ...

Glasgow Coma Scale Eye (ITEMID: 184, UOM: "")
- Raw values: [1, 2, 3, 4]
- Tokens: "184_:1", "184_:2", "184_:3", "184_:4"
```

#### **Special Tokens**
```
<PAD>    # Padding token for sequence alignment (index: 0)
<UNK>    # Unknown token for unseen variables (index: 1)
```

### 2. Token-to-Index Mapping

After tokenization, a vocabulary is created mapping each unique token to an integer index:

```python
token2index = {
    '<PAD>': 0,
    '<UNK>': 1,
    '184_:1': 2,
    '184_:2': 3,
    '220045_bpm:0': 4,
    '220045_bpm:1': 5,
    # ... up to ~40,000 unique clinical tokens
}
```

### 3. Input Data Structure

Each patient's sequence is represented as `(batch_size, max_len, 2)` where:
- **Dimension 0**: Time in hours (0.0, 0.5, 1.0, ..., 48.0)
- **Dimension 1**: Token index (from token2index mapping)

Example for one patient:
```
Time    Token_Index    Meaning
0.5     4              Heart Rate in lowest percentile bin
0.5     1247           Blood pressure in 8th percentile bin  
1.0     4              Heart Rate still in lowest bin
1.5     5              Heart Rate moved to 2nd percentile bin
...     ...            ...
47.5    892            Lab result token
48.0    0              <PAD> token (sequence end)
```

### 4. Embedding Architecture

#### **Token Embedding Layer**
```python
class Embedder(nn.Module):
    def __init__(self, n_tokens, latent_dim, weighted=True):
        self.embedX = Embedding(n_tokens+1, latent_dim, padding_idx=0)
        if weighted:
            self.embedW = Embedding(n_tokens+1, 1)  # Learnable weights
```

- **n_tokens**: Vocabulary size (~40K clinical tokens)
- **latent_dim**: Embedding dimension (typically 32-64)
- **weighted**: Whether to learn importance weights per token

#### **Temporal Aggregation**
Events within each hour are aggregated to create hourly representations:

```python
# For each hour t in [0, 1, 2, ..., 47]:
t_idx = ((t <= T) & (T < t+1)).float()  # Mask events in hour t
if weighted:
    w_t = t_idx * learned_weights        # Apply learned weights
    X_t = w_t * embedded_tokens         # Weight the embeddings
else:
    X_t = t_idx * embedded_tokens       # Simple masking

# Average events within the hour
X_t_avg = X_t.sum(dim=1) / (event_count + 1e-6)
```

### 5. Training Process

#### **Embedding Training**
Embeddings are trained **end-to-end** with the LSTM:

```python
# Forward pass
embedded_sequence = embedder(input_data)  # (batch, 48_hours, latent_dim)
lstm_output = lstm(embedded_sequence)     # (batch, 48_hours, hidden_dim)
predictions = decoder(lstm_output)        # (batch, 48_hours, 1) for dynamic
                                         # (batch, 1) for static

# Loss computation (hourly predictions)
loss = BCE_loss(predictions, mortality_labels)
loss.backward()  # Gradients flow back to embeddings
```

#### **Dynamic vs Static Prediction**
- **Dynamic**: Predict mortality risk at each hour → output shape `(batch, 48, 1)`
- **Static**: Single prediction per patient → output shape `(batch, 1)`

### 6. Prediction Phase

During inference, the same embedding process is used:

1. **Tokenize** new patient's clinical events
2. **Map** tokens to indices using the trained vocabulary
3. **Embed** token indices using learned embedding weights
4. **Aggregate** events hourly using learned attention weights
5. **Process** through trained LSTM and decoder
6. **Output** hourly mortality probabilities

#### **Example Prediction Output**
```
Hour    Mortality_Risk    Clinical_Context
1       0.05             Stable vital signs
6       0.12             Elevated heart rate detected
12      0.23             Blood pressure dropping
18      0.45             Multiple abnormal lab values
24      0.67             Worsening clinical trajectory
```

### 7. Key Advantages

- **No Manual Feature Engineering**: Raw clinical values automatically tokenized
- **Learnable Representations**: Embeddings capture clinical relationships
- **Temporal Awareness**: Hourly aggregation preserves time dynamics
- **Scalable Vocabulary**: Handles ~40K unique clinical concepts
- **Weighted Attention**: Model learns to focus on important events

---

## Pipeline Overview 🚦

> **Python 3.12** is required.

Eight scripts in `pre_processing_scripts/` convert the MIMIC-III CSVs into model-ready NumPy arrays:

| Step | Script | Purpose | Key outputs |
|------|--------|---------|-------------|
| **1** | `1_subject_events.py` | Build cohort, dump raw events & stays **per subject** | `<DATA_ROOT>/<SUBJECT_ID>/stays.csv`<br>`events.csv` |
| **2** | `2_validate_events.py` | Ensure every event maps to a single ICU stay | Cleaned `events.csv` |
| **3** | `3_subject2episode.py` | Split each stay into episodes:<br>• `episode<i>.csv` (static)<br>• `episode<i>_timeseries.csv` (aligned events) | Episode files |
| **4** | `4_truncate_timeseries.py` | Keep first 48 h (or `--t_hours`) and combine `ITEMID_UOM` | `episode<i>_timeseries_48.csv` |
| **5** | `5_split_train_test.py` | Stratified **train / valid / test** split | `splits/*.csv` |
| **6** | `6_generate_value_dict.py` | Decide continuous vs discrete per variable, save raw values | `dictionaries/<t>-values.npy` |
| **7** | `7_quantize_events.py` | Bin continuous vars, tokenise all events, build vocab | Tokenised timeseries + `token2index.npy` |
| **8** | `8_create_arrays.py` | Convert token strings to padded index sequences, save `.npz` | `arrays/<t>_<bins>-arrays.npz` |

---

## Running the Pipeline ⚙️

```bash
# ---- User paths ----
MIMIC_PATH=/path/to/mimic-iii-clinical-database-1.4
DATA_ROOT=/path/to/output_root
SEED=0
T_HOURS=48
N_BINS=20

# 1) Extract raw events + stays
python pre_processing_scripts/1_subject_events.py \
    --mimic3_path "$MIMIC_PATH" \
    --output_path "$DATA_ROOT" \
    --events CHARTEVENTS LABEVENTS OUTPUTEVENTS

# 2) Validate events
python pre_processing_scripts/2_validate_events.py \
    --subjects_root_path "$DATA_ROOT"

# 3) Build episodes
python pre_processing_scripts/3_subject2episode.py \
    --root_path "$DATA_ROOT"

# 4) Truncate to first 48 h
python pre_processing_scripts/4_truncate_timeseries.py \
    --root_dir "$DATA_ROOT" --t_hours $T_HOURS

# 5) Train / valid / test split
python pre_processing_scripts/5_split_train_test.py \
    --root_dir "$DATA_ROOT" --t_hours $T_HOURS --seed $SEED \
    --test_size 0.2 --valid_size 1000

# 6) Generate value dictionary
python pre_processing_scripts/6_generate_value_dict.py \
    --root_dir "$DATA_ROOT" --t_hours $T_HOURS --seed $SEED

# 7) Quantise + tokenise
python pre_processing_scripts/7_quantize_events.py \
    --root_dir "$DATA_ROOT" --t_hours $T_HOURS \
    --n_bins $N_BINS --seed $SEED

# 8) Create final arrays
python pre_processing_scripts/8_create_arrays.py \
    --root_dir "$DATA_ROOT" --t_hours $T_HOURS \
    --n_bins $N_BINS --seed $SEED --max_len 10000


## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eldarzos/Mortality-Prediction-For-ICU-Patients.git
    cd Mortality-Prediction-For-ICU-Patients
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


## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

