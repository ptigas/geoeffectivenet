YEAR=$1
bq query --use_legacy_sql=false \
'CREATE OR REPLACE TABLE fdl-us-geoeffectiveness.supermag.supermag_preprocessed_'$YEAR'
AS
SELECT UNIX_SECONDS(Date_UTC) AS Date_UTC, * EXCEPT (Date_UTC)
FROM `fdl-us-geoeffectiveness.supermag.supermag_preprocessed`
WHERE EXTRACT(YEAR FROM Date_UTC) = '$YEAR';'


echo "> Exporting data to bucket"
gsutil -m rm -r "gs://geo2020_supermag/preprocessed/supermag_preprocessed_$YEAR.*"
bq extract --destination_format=CSV \
    --compression GZIP \
    --print_header=true supermag.supermag_preprocessed_$YEAR \
    "gs://geo2020_supermag/preprocessed/supermag_preprocessed_$YEAR/supermag_preprocessed_$YEAR*.csv.gz"
