from google.cloud import storage
import os

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'seabirdaidatabase-0a68840d87ff.json'


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client.from_service_account_json('seabirdaidatabase-0a68840d87ff.json')
    # storage_client = storage.Client.from_service_account_json(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")



# Example usage
if __name__ == "__main__":
    BUCKET_NAME = 'svea'
    DOWNLOAD_FILE_NAME = 'D20240521-T015305_WBT_741862-15_ES38-7_ES.png'
    DESTINATION_BLOB_NAME = 'D20240521-T015305_WBT_741862-15_ES38-7_ES.png'
    download_blob(BUCKET_NAME, DESTINATION_BLOB_NAME, DOWNLOAD_FILE_NAME)