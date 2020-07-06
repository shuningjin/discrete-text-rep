DATADIR=${1:-'./data'}
ORIGIN=$PWD

# make data directory, exist ok
mkdir -p $DATADIR
cd $DATADIR

# agnews
gdown https://drive.google.com/uc?id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms -O ag_news_csv.tar.gz
# dbpedia
gdown https://drive.google.com/uc?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k -O dbpedia_csv.tar.gz
# yelp review full
gdown https://drive.google.com/uc?id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0 -O yelp_review_full_csv.tar.gz

tar -xvzf ag_news_csv.tar.gz
tar -xvzf dbpedia_csv.tar.gz
tar -xvzf yelp_review_full_csv.tar.gz

# delete files
rm -r ag_news_csv.tar.gz dbpedia_csv.tar.gz yelp_review_full_csv.tar.gz

cd $ORIGIN
