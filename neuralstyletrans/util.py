import boto3
import os

def create_tmp_path():
    """
    Create temporary path for storing images
    """
    img_dir = '/tmp/picasso'
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

def upload_to_s3(image):
    """
    Connect to s3 instance and upload the transformed image.
    Return the presigned url.
    """
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(name='picasso-lambdaschool')

    # Upload a file
    upload_key = 'deeptransform/'+image
    s3_bucket.upload_file(Filename=image, Key=upload_key)

    # Get url
    s3_client = boto3.client('s3')
    url = s3_client.generate_presigned_url('get_object',
            Params = {'Bucket': 'picasso-lambdaschool', 'Key': upload_key},
            ExpiresIn = 100)

    return url
