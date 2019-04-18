from flask import Flask, request
from flask_celery import make_celery
from decouple import config
from deeptransformimpl import trigger_deeptransform
from flask_json import FlaskJSON, JsonError, json_response
import requests

# Create and configure an instance of the Flask application.
app = Flask(__name__)
FlaskJSON(app)

# Fetch environment configurations
DEEPAI_KEY = config('DEEPAI_KEY')
app.config['CELERY_BROKER_URL'] = config('CELERY_BROKER_URL')
app.config['CELERY_RESULT_BACKEND'] = config('CELERY_RESULT_BACKEND')

# Initialize celery app with flask app
celery = make_celery(app)

@app.route('/fasttransform', methods=['POST'])
def fast_neural_style_transform():
    """
    For fast neural style transformation of image we are using API
    exposed by deepai.org
    """

    data = request.get_json(force=True)
    try:
        key = data['request_key']
        style_url = data['style_url']
        content_url = data['content_url']
    except (KeyError):
        raise JsonError(description='Key Error: Key missing in the request')

    # Invoke deepai.org neural-style API

    deepai_resp = requests.post(
    "https://api.deepai.org/api/neural-style",
    data={
        'style': style_url,
        'content': content_url,
    },
    headers={'api-key': DEEPAI_KEY}
    )

    deepai_resp_json = deepai_resp.json()

    try:
        tranformed_image_url = deepai_resp_json['output_url']
    except (KeyError):
        err_msg = 'Trans Error: Neural style transformation failed'
        raise JsonError(status_=500, request_key=key, description=err_msg)

    return json_response(request_key=key,
            output_url=tranformed_image_url)

@app.route('/deeptransform', methods=['POST'])
def deep_neural_style_transform():
    """
    For deep neural style transformation of image we are using API
    developed internally by lambda school students
    """

    data = request.get_json(force=True)
    try:
        key = data['request_key']
        style_url = data['style_url']
        content_url = data['content_url']
    except (KeyError):
        raise JsonError(description='Key Error: Key missing in the request')

    deeptransform_async.delay(key, style_url, content_url)

    return json_response(request_key=key,
            description='Deep transformation is in progress')

@celery.task(name='app.deeptransform_async')
def deeptransform_async(key, style_url, content_url):
    trigger_deeptransform(key, style_url, content_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

