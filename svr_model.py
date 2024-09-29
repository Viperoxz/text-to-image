from flask import Flask, request, render_template
from text2img_model import create_pipeline, text2img

app = Flask(__name__)

IMAGE_PATH = "static/output.jpg"

pipeline = create_pipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        #Tra ve giao dien web
        return render_template('index.html')
    else:
        #Xu ly submit prompt -> image -> tra ve
        user_input = request.form['prompt']
        print('Start generating image...')
        img = text2img(user_input, pipeline)
        print('Done generating image')
        img.save(IMAGE_PATH)

        return render_template('index.html', image_url=IMAGE_PATH)
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888, use_reloader=False)
