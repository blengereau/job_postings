Lancer la commande sur un fichier parquet avec une colonne description: 

python predict_job_features.py ^
  --model-dir moderncamembert_job_features_17_labels ^
  --input job_postings.parquet ^
  --output job_postings_predicted.parquet ^
  --text-col description