import pandas as pd

from utils import RESULTS_PATH, DTYPES

for f in (RESULTS_PATH / "results").glob("e[14]-daiba17.csv"):
    data: pd.DataFrame = pd.read_csv(f, dtype=DTYPES) # type: ignore
    data = data[[_ for _ in data.columns if not _.startswith("Unnamed") and not _ =="index"]]
    data.to_csv(f, index=False)
