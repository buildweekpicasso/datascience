# This is an implementation class for deep neural style transformation 

def trigger_deeptransform(key, style_url, content_url):
    """
    This is the starting point of deep neural style transformation
    """
    print('Inside trigger_deeptranform')
    print(f'key: {key}')
    print(f'style_url: {style_url}')
    print(f'content_url: {content_url}')

def trigger_deeptransform_notification(status, key, output_url):
    """
    This is the end point of deep neural style transformation
    """
    print('Inside trigger_deeptransform_response')
    print(f'status: {status}')
    print(f'key: {key}')
    print(f'output_url: {output_url}')

