For stitching.py

stitching.py takes in three video input and stitches them to produce a panorama
video.

Users can choose to stitch by using the clicking function to obtain good points for homography, or he can choose to compute the homography automatically by uncommenting the getHomography functions that are called inside.

For trackAndMapBirdEyeView.py

This file will take in the stitched video, the background and a image of a football field to process. The program will track the players, goalkeepers and referees and output as animation in the football field image provided. The objects are represented with color nodes as shown in the table below.

Role        | Team 1 | Team 2|
------------------------------
Players     |  blue  |  red  |
------------------------------
Goalkeeper  |  green | white |
------------------------------
Referee     |     yellow     |
------------------------------

Users have to ensure that stitchedVideo.mov, football_field.jpg, background.jpg are in the same folder.

Software required : 
python 2.7.10 
opencv 2.4
numpy