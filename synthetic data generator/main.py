import os
import numpy as np

from data_loader  import load_dataset
from preprocessor import prepare_training_data
from train        import train
from generate     import generate


def main():
    print("=" * 60)
    print("  TimeGAN  —  Battery C/10 Discharge Curve Synthesis")
    print("=" * 60)

    # Step 1: load raw data from the Zenodo download
    print("\n[1/4]  Loading dataset ...")
    records = load_dataset()

    # Step 2: resample every curve to 100 uniform Q-steps, normalise
    print("\n[2/4]  Preprocessing ...")
    data_norm, params, meta = prepare_training_data(records)

    # Quick sanity check before spending hours on training
    assert data_norm.ndim == 3,      "Expected shape (N, T, F)"
    assert data_norm.shape[1] == 100, "Expected 100 time steps"
    assert np.all(data_norm >= 0) and np.all(data_norm <= 1), \
        "Normalised data should be in [0, 1]"

    # Save the processed array so you can re-run training without re-parsing
    np.save(os.path.join("processed", "data_norm.npy"),  data_norm)
    np.save(os.path.join("processed", "meta.npy"),       np.array(meta))
    print(f"Processed data saved to 'processed/'.")

    # Step 3: train TimeGAN (all three phases)
    print("\n[3/4]  Training TimeGAN ...")
    train(data_norm)

    # Step 4: generate 200 000 synthetic cells
    print("\n[4/4]  Generating synthetic data ...")
    generate()

    print("\nAll done.")


if __name__ == "__main__":
    main()
