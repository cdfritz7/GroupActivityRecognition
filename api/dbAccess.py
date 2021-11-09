import pymongo
import datetime
from bson.objectid import ObjectId
from bson.int64 import Int64

'''
-----------------
Helper Functions
-----------------
'''
def cullTimedOutInteractions(db, areaId):
    area = db.area.find_one({"_id":ObjectId(areaId)})

    if(area == None):
        return False

    expirationTime = area["expirationTime"]
    milliSince1970 = datetime.datetime.now().timestamp() * 1000

    db.areaUserInteraction.delete_many({
        "epochMilliseconds": {"$lte" : milliSince1970 - expirationTime}
    })

    return True

def voteAreaLabel(db, areaId):

    area = db.area.find_one({"_id":ObjectId(areaId)})

    if(area == None):
        return None

    labels = list(db.areaUserInteraction.aggregate([
                                                {"$match" : {"areaId":areaId}},
                                                {"$sortByCount": "$label"}
                                                  ]))


    if(len(labels) == 0):
        return None

    return {'label':labels[0]['_id'], 'count':labels[0]['count']}

def cullReplacedInteractions(db, areaId, userId):
    db.areaUserInteraction.delete_many({
        "areaId":str(areaId),
        "userId":str(userId)
    })

def createNewAreaUserInteraction(db, areaId, userId, label):

    milliSince1970 = datetime.datetime.now().timestamp() * 1000

    db.areaUserInteraction.insert_one({
        "label":str(label),
        "epochMilliseconds":float(milliSince1970),
        "areaId":str(areaId),
        "userId":str(userId)
    })

def findArea(db, lat, lng):
    #lat lon should be in decimal format with north and east being positive
    return db.area.find({
                "topLeftLat": {"$gte":float(lat)},
                "topLeftLng": {"$lte":float(lng)},
                "bottomRightLat": {"$lte":float(lat)},
                "bottomRightLng": {"$gte":float(lng)}
            })

'''
-----------------
Main Functions Called from API
-----------------
'''
def sendActivityFromUser(db, lat, lng, userId, label):

    areas = list(findArea(db, lat, lng))

    if(len(areas) == 0):
        return "failure"

    for area in areas:
        cullReplacedInteractions(db, area['_id'], userId)
        createNewAreaUserInteraction(db, area['_id'], userId, label)

    return "success"

def getAreaActivity(db, areaId):

    if not cullTimedOutInteractions(db, areaId):
        return None

    return voteAreaLabel(db, areaId)

def addArea(db, topLeftLat, topLeftLng, bottomRightLat, bottomRightLng, expirationTime):

    print(db)
    print(topLeftLat)
    
    ret = ""

    try:
        db.area.insert_one({
            "topLeftLat": float(topLeftLat),
            "topLeftLng": float(topLeftLng),
            "bottomRightLat": float(bottomRightLat),
            "bottomRightLng": float(bottomRightLng),
            "expirationTime": Int64(expirationTime)
        })
        ret = "success"

    except:
        ret = "failure"

    return ret
