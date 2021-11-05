from flask import Flask, request
from flask_cors import CORS
import pymongo
import urllib
import dbAccess
from bson.json_util import dumps

username = urllib.parse.quote("user")
password = urllib.parse.quote("xmv5IFdUoDDNwhJh")
databaseName = urllib.parse.quote("GroupActivityRecognition")
connectionStr = f"mongodb+srv://{username}:{password}@cluster0.ysnpp.mongodb.net/{databaseName}?retryWrites=true&w=majority"
client = pymongo.MongoClient(connectionStr)

app = Flask(__name__)
db = client[databaseName]

@app.route('/getAllAreas/', methods=['GET', 'POST'])
def getAreas():
    return dumps(list(db.area.find()))


@app.route('/sendActivityFromUser/', methods=['POST'])
def sendActivityFromUser():
    lat = float(request.form['lat'])
    lng = float(request.form['lng'])
    userId = request.form['userId']
    label = request.form['label']

    return dbAccess.sendActivityFromUser(db, lat, lng, userId, label)


@app.route('/getAreaActivity/<areaId>', methods=['GET'])
def getAreaActivity(areaId):
    return str(dbAccess.getAreaActivity(db, areaId))


@app.route('/addArea/', methods=['POST'])
def addArea():
    topLeftLat = float(request.form['topLeftLat'])
    topLeftLng = float(request.form['topLeftLng'])
    bottomRightLat = float(request.form['bottomRightLat'])
    bottomRightLng = float(request.form['bottomRightLng'])
    expirationTime = int(request.form['expirationTime'])

    return dbAccess.addArea(db, topLeftLat, topLeftLng, bottomRightLat,
                            bottomRightLng, expirationTime)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
