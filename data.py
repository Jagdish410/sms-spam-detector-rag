"""Download and load the SMS Spam Collection dataset (UCI)."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PATH = Path("data/SMSSpamCollection")


def load_data() -> pd.DataFrame:
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print("Downloading dataset from UCI...")
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            DATA_PATH.write_bytes(z.read("SMSSpamCollection"))
        except Exception as e:
            raise SystemExit(
                f"Could not download the dataset ({e}).\n"
                "Download 'SMS Spam Collection' manually from UCI or Kaggle and place the "
                "tab-separated file at data/SMSSpamCollection (columns: label<TAB>text)."
            )
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])
    df["y"] = (df["label"] == "spam").astype(int)
    return df
