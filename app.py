import os
from PIL import Image
from matplotlib import pyplot as plt
from flask import Flask,render_template

root = r'C:\Users\firat\OneDrive\Masaüstü\cat-dog-bird-classifier\dataset\test\cat'

resimler = os.listdir(root)
#print("resimler: ",resimler)
print()

# for i in range(10):
#     print(resimler[i])
    
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/contact")
def contact():
    return render_template("contact.html")


app.run(debug=True)