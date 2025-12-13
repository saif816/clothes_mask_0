import boto3
from botocore.exceptions import NoCredentialsError
from io import BytesIO

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIA4MTWK4OCGGW36VHX',
    aws_secret_access_key='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',
    region_name="eu-north-1"
)

# Define the bucket name and file details
bucket_name = 'fitcheck-assets'
local_file_path = 'images.jpeg'  # Replace with your file path
s3_file_key = f'renderss/{local_file_path}'  # This is the S3 object name

try:
    # Open the local file in binary mode
    with open(local_file_path, 'rb') as file_data:
        # Upload the file to S3
        s3_client.upload_fileobj(
            file_data,
            bucket_name,
            s3_file_key,
            ExtraArgs={'ContentType': 'image/png'}  # Publicly readable
        )

    # Generate the public URL
    object_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_key}"
    print(f"File uploaded successfully. Public URL: {object_url}")

except FileNotFoundError:
    print("The local file was not found.")
except NoCredentialsError:
    print("Credentials not available.")
except Exception as e:
    print(f"An error occurred: {e}")


# https://fitcheck-assets.s3.amazonaws.com/renderss/1test.png
# https://fitcheck-assets.s3.amazonaws.com/renderss/test2.png