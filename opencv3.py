import cv2
from emot.nnModel import EMR

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier("haarcascade_files/nariz.xml")
curr_emotion = 'happy'


def brighten(data, b):
    datab = data * b
    return datab


def kyaEmoji(emotion):
    return 'emojis/{}.png'.format(emotion)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if not len(faces) > 0:
        return None
    for (x, y, w, h) in faces:
        emoji = cv2.imread(kyaEmoji(curr_emotion), -1)
        # We load the emoji image with -1 (negative one) as the second parameter to load all the layers in the image
        #  The image is made up of 4 layers (or channels): Blue, Green, Red, and an Alpha transparency layer (knows as BGR-A).
        #  The alpha channel tell us which pixels in the image should be transparent (show the image underneath) or should be
        # non-transparent (made up of a combination of the other 3 layers
        try:
            emoji.shape[2]
        except:
            emoji = emoji.reshape(emoji.shape[0], emoji.shape[1], 1)
        orig_mask = emoji[:, :, 3]
        # Here we take just the alpha layer and create a new single-layer image that we will use for masking.
        ret1, orig_mask = cv2.threshold(orig_mask, 10, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)
        # We take the inverse of our mask. The initial mask will define the area for the emoji,
        # and the inverse mask will be for the region around the emmoji).
        emoji = emoji[:, :, 0:3]
        # Here we convert the emoji1 image to a 3-channel BGR image (BGR rather than BGR-A is
        # required when we overlay the emoji1 image over the webcam image later).

        origemoji1Height, origemoji1Width = emoji.shape[:2]

        roi_gray = image[y:y + h, x:x + w]
        # We create a greyscale ROI for the area where the face was discovered
        # (remember that we will be looking for a nose within this face, and Haar cascade classifiers operate on greyscale images.
        roi_color = frame[y:y + h, x:x + w]

        # We also keep a color ROI for the area where the face is, as we will draw our emoji1 over the color ROI.
        # (Remember that the x and y co-ordinates are backwards when selecting a ROI.

        # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in nose:
            # Un-comment the next line for debug (draw box around the nose)
            # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

            # The emoji size can be adjusted here
            emoji1Width = 30 * nw
            emoji1Height = emoji1Width * origemoji1Height / origemoji1Width

            # Center the emoji1 on the bottom of the nose
            x1 = nx - (emoji1Width / 4)
            x2 = nx + nw + (emoji1Width / 4)
            y1 = ny + nh - (emoji1Height / 2)
            y2 = ny + nh + (emoji1Height / 2)

            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            # Re-calculate the width and height of the emoji1 image
            emoji1Width = (x2 - x1)
            emoji1Height = (y2 - y1)

            # Re-size the original image and the masks to the emoji1 sizes
            # calcualted above
            emoji1 = cv2.resize(emoji, (emoji1Width, emoji1Height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (emoji1Width, emoji1Height), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (emoji1Width, emoji1Height), interpolation=cv2.INTER_AREA)

            # take ROI for emoji1 from background equal to size of emoji1 image
            roi = roi_color[y1:y2, x1:x2]

            # roi_bg contains the original image only where the emoji1 is not
            # in the region that is the size of the emoji1.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # roi_fg contains the image of the emoji1 only where the emoji1 is
            roi_fg = cv2.bitwise_and(emoji1, emoji1, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst

            break
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(frame, str(curr_emotion), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image


network = EMR()  # emotion neural network
network.build_network()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()
    # Predict result with mera NN
    face = format_image(frame)
    result = network.predict(face)
    # Write results in frame
    if result is not None:
        aa = list(result[0])
        # face_image = feelings_faces[aa.index(max(aa))]
        curr_emotion = EMOTIONS[aa.index(max(aa))]
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
