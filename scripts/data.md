


#### summary

Dataset | # Classes | Train | Dev | Test
:--- | ---: | ---: | ---: | ---:
AG News | 4 | 115K | 5K | 7.6K
DBPedia | 14 | 555K | 5K | 70K
Yelp Review Full | 5 | 645K | 5K | 50K


#### data directory structure
(assume the directory is `cls_data_dir`)

- after download

  ```bash
  # data dir structure (after download)
  tree cls_data_dir

  cls_data_dir
  ├── ag_news_csv
  │   ├── classes.txt
  │   ├── readme.txt
  │   ├── test.csv
  │   └── train.csv
  ├── dbpedia_csv
  │   ├── classes.txt
  │   ├── readme.txt
  │   ├── test.csv
  │   └── train.csv
  └── yelp_review_full_csv
      ├── readme.txt
      ├── test.csv
      └── train.csv

  ```



- after train_dev_split

  ```bash
  # data dir structure (after train_dev_split)
  # split train.csv to train.dev.csv and train.train.csv
  # our experiment use:
  #     train.train.csv (train), train.dev.csv (dev), test.csv (test)

  tree cls_data_dir

  cls_data_dir
  ├── ag_news_csv
  │   ├── classes.txt
  │   ├── readme.txt
  │   ├── test.csv
  │   ├── train.csv
  │   ├── train.dev.csv
  │   └── train.train.csv
  ├── dbpedia_csv
  │   ├── classes.txt
  │   ├── readme.txt
  │   ├── test.csv
  │   ├── train.csv
  │   ├── train.dev.csv
  │   └── train.train.csv
  └── yelp_review_full_csv
      ├── readme.txt
      ├── test.csv
      ├── train.csv
      ├── train.dev.csv
      └── train.train.csv

  ```

- line count

  ```bash
  # line count
  wc -l cls_data_dir/*/*

   4 cls_data_dir/ag_news_csv/classes.txt
  19 cls_data_dir/ag_news_csv/readme.txt
  7600 cls_data_dir/ag_news_csv/test.csv
  120000 cls_data_dir/ag_news_csv/train.csv
  5000 cls_data_dir/ag_news_csv/train.dev.csv
  115000 cls_data_dir/ag_news_csv/train.train.csv

  14 cls_data_dir/dbpedia_csv/classes.txt
  17 cls_data_dir/dbpedia_csv/readme.txt
  70000 cls_data_dir/dbpedia_csv/test.csv
  560000 cls_data_dir/dbpedia_csv/train.csv
  5000 cls_data_dir/dbpedia_csv/train.dev.csv
  555000 cls_data_dir/dbpedia_csv/train.train.csv

  16 cls_data_dir/yelp_review_full_csv/readme.txt
  50000 cls_data_dir/yelp_review_full_csv/test.csv
  650000 cls_data_dir/yelp_review_full_csv/train.csv
  5000 cls_data_dir/yelp_review_full_csv/train.dev.csv
  645000 cls_data_dir/yelp_review_full_csv/train.train.csv
  ```
