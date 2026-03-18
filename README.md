```bash
python .\extract_grouped.py --interval-ms 1000 --clean; python .\expand_dataset_offline.py --source-subdir extracted_grouped --output-subdir extracted_group_augmented --target-fold 15.0 --clean --equalize-classes
```