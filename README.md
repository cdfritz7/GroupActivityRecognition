
# GroupActivityRecognition

## API Documentation

Additional documentation for deploying API found in /api

### Database Structure

There are two collections in the database : ```area``` and ```areaUserInteraction```
Documents in these collections should have the following fields

```area```
 -  _id (ObjectId) : a unique identifier for this document
  - topLeftLat (Double) : latitude corresponding to the top left corner of this area's bounding box
  - topLeftLng (Double) : longitude corresponding the the top left corner of this area's bounding box
  - bottomRightLat (Double) : latitude corresponding to the bottom right corner of this area's bounding box
  - bottomRightLng (Double) : longitude corresponding the the bottom right corner of this area's bounding box
  - expirationTime (Int64)  : time in milliseconds after which corresponding user labels will expire

```areaUserInteraction```
  - _id (ObjectId) : a unique identifier for this document
  - label (String) : the label for this interaction
  - epochMilliseconds (Double) : number of milliseconds between this objects creation and Jan 1st 1970
  - areaId (String) : the _id of the area to which this document corresponds
  - userId (String) : a unique identifier for the user to which this document corresponds


### Routes

/getAllAreas
  - accepts GET and POST requests
  - returns all areas

/sendActivityFromUser
  - accepts POST requests with the fields "lat", "lng", "userId" and "label"
  - finds any areas corresponding to the user's lat and lng
  - if that user x area pair already has an interaction, that interaction is deleted
  - adds a new areaUserInteraction for each area that overlaps with the users current location

/getAreaActivity/\<areaId>
  - accepts GET requests
  - removes any expired areaUserInteractions associated with the area that has _id == areaId
  - returns : the most common label for all areaUserInteractions associated with the area that has
    _id == areaId, the number of occurrences of the most common label, and the total number of
    areaUserInteractions for the corresponding area

/addArea
  - accepts POST requests with the fields "topLeftLat", "topLeftLng", "bottomRightLat",
    "bottomRightLng", and "expirationTime"
  - inserts an area into the database
