from flask import Flask, request
from flask.templating import render_template
import os
import time
from detection import *

global detector
detector = init()

app = Flask(__name__)
app.debug = False

# Main page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index')
def return_index():
    return render_template('index.html')


@app.route('/det_cat')
def nst_cat():
    return render_template('det_cat.html')


@app.route('/det_tree')
def nst_tree():
    return render_template('det_tree.html')


@app.route('/det_life')
def nst_life():
    return render_template('det_life.html')


@app.route('/det_post', methods=['GET', 'POST'])
def nst_post():
    if request.method == 'POST':
        now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        dest = os.path.join("static/temporary", now)
        os.mkdir(dest)

        user_img = request.files['user_img']
        user_img_path = os.path.join(dest, str(user_img.filename))
        user_img.save(user_img_path)

        option = request.form['options']

        if option == "cat":
            path_details, path_detected, path_plot, result_str = test_cat(detector['cat'], user_img_path)

            return render_template('det_post.html', user_img=path_detected, user_url=path_plot, user_result=result_str,
                                   my_tr_list=path_details.keys(), my_td_list=path_details.values())
        if option == "tree":
            path_details, path_detected, final_sentence = test_tree(detector['tree'], user_img_path)

            return render_template('det_post.html', user_img=path_detected, user_result=final_sentence,
                                   my_tr_list=path_details.keys(), my_td_list=path_details.values())
        if option == "life":
            results = test_life(detector['life'], user_img_path)
            return render_template('det_post.html', my_tr_list=results.keys(), my_td_list=results.values())

if __name__ == "__main__":
    # get port. Default to 8080
    port = int(os.environ.get('PORT', 8080))

    # run app
    app.run(host='0.0.0.0', port=port)