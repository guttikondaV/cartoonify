import os

import src.env  # This module has side-effects and no exports
import src.etl as etl
import src.train as train
import src.infer as infer


def main():
    """
    The main function of the package.

    This function extracts the data, loads it, trains the model and then
    runs inference on the test data.

    Use only this function to run the package.
    """
    etl.extract_data()
    etl.load_data(100, 5)

    train.train()

    infer.infer(os.path.join(src.env.PROCESSED_DATA_DIR, "people_faces/test"))


if __name__ == "__main__":
    main()
