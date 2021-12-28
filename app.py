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


@app.route('/det_get')
def nst_get():
    return render_template('det_get.html')


@app.route('/det_get2')
def nst_get2():
    return render_template('det_get2.html')


@app.route('/det_post', methods=['GET', 'POST'])
def nst_post():
    if request.method == 'POST':
        # User Image (target image)
        now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        dest = os.path.join("static/temporary", now)
        os.mkdir(dest)

        user_img = request.files['user_img']
        user_img_path = os.path.join(dest, str(user_img.filename))
        user_img.save(user_img_path)

        # for radio button
        option = request.form['options']

        # args['in_buf'].put((user_img_path, option))
        # while args['out_buf'].empty():
        #     pass

        if option == "tree":
            path_details, path_detected, final_sentence = test_tree(detector['tree'], user_img_path)

            return render_template('det_post.html', user_img=path_detected, user_result=final_sentence,
                                   my_tr_list=path_details.keys(), my_td_list=path_details.values())

        elif option == "cat":
            path_details, path_detected, path_plot, result_str = test_cat(detector['cat'], user_img_path)

            return render_template('det_post.html', user_img=path_detected, user_url=path_plot, user_result=result_str,
                                   my_tr_list=path_details.keys(), my_td_list=path_details.values())

if __name__ == "__main__":
    # get port. Default to 8080
    port = int(os.environ.get('PORT', 8080))

    # run app
    app.run(host='0.0.0.0', port=port)