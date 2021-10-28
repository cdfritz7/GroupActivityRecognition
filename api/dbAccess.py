import pymongo

def cullTimedOutInteractions(db, areaId):
    pass

def cullReplacedInteraction(db, areaId, userId):
    pass

#lat lon should be in decimal format with north and east being positive
def findArea(db, lat, lng):

    return db.area.find({
                "topLeftLat": {"$gte":lat},
                "topLeftLng": {"$lte":lng},
                "bottomRightLat": {"$lte":lat},
                "bottomRightLng": {"$gte":lng}
            })

def sendActivityFromUser(db, lat, lng, userId, label):
    #find associated area if any, return false if no matching area found
    area = findArea(db, lat, lng)
    return False

    #remove any areaUserInteraction matching both the area and the user

    #create new areaUserInteraction

def getAreaActivity(db, areaId):

    #find area expiration time

    #remove any areaUserInteractions belonging to the area that are past the
    #expiration time
    cullTimedOutInteractions(db, areaId)

    #from remaining interactions, find the most common
