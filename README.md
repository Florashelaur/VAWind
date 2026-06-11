# VAWind

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- data_provider/
|-- exp/
|-- models/
|-- scripts/
|   `-- run_experiments.sh
|-- utils/
|-- requirements.txt
`-- run.py
```

Dataset :See [`data/README.md`](data/README.md) for the ERA5 source, variables, CEEMDAN
requirements, and CSV schema.

## Installation

Python 3.9 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Training

Train the complete VAWind model with TCN for 10 m wind-speed forecasting:

```bash
python run.py \
  --model TCN \
  --root_path ./data \
  --data_path zhongshan_10m.csv \
  --target wind_speed_10m \
  --pred_len 12 \
  --use_noise 1
```M
```

Available horizons are 3, 6, 9, 12 

For 100 m wind-speed forecasting:

```bash
python run.py \
  --model TCN \
  --root_path ./data \
  --data_path zhongshan_100m.csv \
  --target wind_speed_100m \
  --pred_len 12 \
  --use_noise 1
```
