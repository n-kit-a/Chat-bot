import pickle
import ssl
import pandas as pd
import sister
from flask import Flask
from flask import request

app = Flask(__name__)

ssl._create_default_https_context = ssl._create_unverified_context
embedder = sister.MeanEmbedding(lang="en")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route("/get_answer", methods=["POST"])
def get_answer():

    content = request.get_json()
    user_replica = pd.DataFrame(embedder(content['user_replica'].lower()))
    pred = loaded_model.predict(user_replica.T)

    response = {
        '1': 'Hello',
        '2': 'Goodbye',
        '3': 'you are welcome',
        '4': 'there must be time API',
        '5': 'i can tell y the time, tell y the weather or recommend resources for programming',
        '6': 'y can find a lot of courses for programing there: stepik.org',
        '7': 'watch dlcourse.ai',
        '8': 'there must be weather API'
    }

    res = response[str(pred[0])]


    return {"response": res}


if __name__ == '__main__':
    app.run()
