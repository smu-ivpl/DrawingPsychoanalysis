import os
import time

from flask import Flask, request
from flask.templating import render_template
# from detection_old import Detector, ImageQue, detect
from threading import Thread, Event

app = Flask(__name__)
app.debug = False

args = {}

args['running'] = Event()
args['stop'] = Event()
args['in_buf'] = ImageQue()
args['out_buf'] = ImageQue()
args['detector'] = Detector()

th = Thread(target=detect, kwargs=args)
th.start()

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

        args['in_buf'].put((user_img_path, option))
        while args['out_buf'].empty():
            pass

        if option == "tree":
            # det = detection.Detector()
            # for tree test
            # detected_branch, detected_trunk, detected_root, detected_tree_img, tree_result = det.test_tree(user_img_path)
            (detected_branch, detected_trunk, detected_root, detected_tree_img, tree_result) = args['out_buf'].get()

            return render_template('det_post.html', user_img=detected_tree_img, user_result=tree_result, user_url=""
                                   , th_1='Branches', th_2='Trunk', th_3='Root',
                                   td_1=detected_branch, td_2=detected_trunk, td_3=detected_root)

        elif option == "cat":
            # det = detection.Detector()
            # for cat test
            # detected_cat, detected_head, detected_body, detected_cat_img, cat_result, cat_plot = det.test_cat(user_img_path)
            result = args['out_buf'].get()
            print(result)
            (detected_cat, detected_head, detected_body, detected_cat_img, cat_result, cat_plot) = result
            return render_template('det_post.html', user_img=detected_cat_img, user_result=cat_result, user_url=cat_plot
                                   , th_1='Cat', th_2='Head', th_3='Body',
                                   td_1=detected_cat, td_2=detected_head, td_3=detected_body)

# if __name__ == "__main__":
#     # get port. Default to 8080
#     port = int(os.environ.get('PORT', 8080))
#
#     # run app
#     app.run(host='0.0.0.0', port=port)
