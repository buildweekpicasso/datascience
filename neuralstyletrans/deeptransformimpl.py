# This is an implementation class for deep neural style transformation 
from .neuralstyle import deeptransform
import requests
import urllib.request
from .util import create_tmp_path

def trigger_deeptransform(key, style_url, content_url):
    """
    This is the starting point of deep neural style transformation
    """
    # Check and create temp file path
    create_tmp_path()

    download_path = '/tmp/picasso/'
    style_image_path = download_path+key+'style.jpg'
    content_image_path = download_path+key+'content.jpg'

    urllib.request.urlretrieve(style_url, style_image_path)
    urllib.request.urlretrieve(content_url, content_image_path)

    deeptransform(key, style_image_path, content_image_path)


def trigger_deeptransform_notification(status, key, output_url):
    """
    This is the end point of deep neural style transformation
    """
    print('Inside trigger_deeptransform_response')
    print(f'status: {status}')
    print(f'key: {key}')
    print(f'output_url: {output_url}')

