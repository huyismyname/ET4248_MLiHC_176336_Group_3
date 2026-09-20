.PHONY: setup data eda baseline eval strip clean

setup:
	pip install -r requirements.txt

data:
	python scripts/download_data.py --dest ./data

eda:
	python -m src.eda --config configs/default.yaml

baseline:
	python -m src.train --config configs/default.yaml

eval:
	python -m src.evaluate --config configs/default.yaml

# Bỏ output khỏi notebook trước khi commit (giữ repo sạch)
strip:
	nbstripout notebooks/*.ipynb

clean:
	rm -rf outputs/ wandb/ __pycache__ src/__pycache__
