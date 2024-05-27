# Svea
This folder includes scripts and instructions to run real-time processing and analysis of echosounder data onboard R/V Svea. 

## Configure the scripts
##### Create folders to store output data
```
mkdir out/img
mkdir out/csv
sudo nano logfile.log # create logfile
```

##### Configure database and cloud storage in Google
The communication and data storage in this repo is based on Google Cloud Services. Before running this project, make sure that you have configured the following:
* Cloud SQL
* Cloud Storage Bucket
* IAM Role with permissions to upload to the Cloud Services
* JSON credentials-file accessible in the repo (with updated path)

##### Modify the bash script
`svea.sh` is the bash script that controls the entire process onboard R/V Svea. Make sure that all paths are updated based on the configuration on your hardware. 

##### Configure a cronjob to schedule the script to run in a given interval
```
crontab -e
```
Add the following row to your crontab file:
```
*/10 * * * * /PATH/TO/REPO/echoedge/edge/svea/svea.sh # to run every 10th minute
```
Restart cron and verify that everything is working
```
sudo systemctl restart cron
sudo systemctl status cron
```
