## Acknowledgment
This codebase is heavily based on and modified from the highly successful **[MEGA repository](https://github.com/facebookresearch/mega/blob/main/examples/mega/README.lra.md)**. We sincerely thank the authors of MEGA for their excellent open-source framework.

## Repository Structure

To ensure complete reproducibility, we provide the following core assets:
* `checkpoint/`: Contains the pre-trained model weights for various tasks.
* `out_log/`: Contains the complete training logs. These logs explicitly record the exact hyperparameter settings and configurations used for each corresponding task.
* `run_lra/`: Contains the executable shell scripts (`.sh`) for training and evaluation on the Long Range Arena (LRA) benchmark.
* `fairseq/`: Contains the core implementation of the Prefixformer architecture and modified fairseq modules.

## LRA Data Preparation

Before running the scripts, please download the processed LRA datasets. 
Download the [processed data here](https://dl.fbaipublicfiles.com/mega/data/lra.zip) (provided by the MEGA repository). 
*(Note: The original raw data is from the [Google LRA repo](https://github.com/google-research/long-range-arena)).*

Extract the downloaded `lra.zip` to a directory on your machine.

## How to Use

You can easily train or evaluate the model by running the scripts provided in the `run_lra/` directory (e.g., `image.sh`, `listops.sh`, `text.sh`, etc.).

**Important Path Configuration:**
Before executing any `.sh` file, please open it and modify the following environment variables to match your local paths:

1.  `DATA`: Point this to your downloaded and extracted LRA dataset directory. 
    * *Example:* Change `DATA=/workspace/lra/listops` to `DATA=/your/local/path/lra/listops`
2.  `SAVE_ROOT`: Point this to the directory where you want to save outputs or load checkpoints.
    * *Example:* Change `SAVE_ROOT=/workspace/lra_saved/listops` to `SAVE_ROOT=/your/local/path/Prefixformer/checkpoint/`

Run the script:
```bash
cd run_lra
bash listops.sh
```

## What to do if you encounter errors (Alternative Execution)

If you encounter any errors during training or inference when running this repository directly, you can easily resolve them by integrating our core files into the original MEGA repository.

Simply follow these 3 steps:

1. Copy `fairseq/models/lra/prefixformer_lra_encoder.py` from this repo and paste it into MEGA's `fairseq/models/lra/` directory.
2. Copy `fairseq/modules/prefixformer_sentence_encoder_layer.py` from this repo and paste it into MEGA's `fairseq/modules/` directory.
3. Copy `fairseq/models/lra/model.py` from this repo and **replace** the existing `model.py` in MEGA's `fairseq/models/lra/` directory.

After replacing these specific files, you can directly run the models using MEGA's original pipeline without any issues.
