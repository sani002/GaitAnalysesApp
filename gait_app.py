import streamlit as st
from pyexpat import model
import mediapipe as mp
import cv2
import keras
import numpy as np
import tempfile
import time
from PIL import Image

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                            content:'This app is in its early stage.';
                            visibility: visible;
                            display: block;
                            position: relative;
                            #background-color: red;
                            padding: 5px;
                            top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Required functions


def multiple_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    res = results.pose_landmarks.landmark
    pose = np.array([[res[i].x, res[i].y, res[i].z, res[i].visibility] for i in [
                    0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 31, 32]]).flatten() if res else np.zeros(68)
    return pose


no_sequences = 1200
sequence_length = 130
actions = np.array(['Antalgic gait', 'Lurch gait',
                   'Normal gait', 'Stiff legged gait', 'Trendelenburg gait'])
label_map = {label: num for num, label in enumerate(actions)}
model = keras.models.load_model('action.h5')

st.title('Gait Analyzer 1.0')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Come on! Do you walk right?')
st.subheader('Lets check')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.selectbox('Choose the App mode',
                        ['About', 'LessDO IT']
                        )

if app_mode == 'About':
    st.markdown('This app is still a bay-beh. Have some mercy!')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "![Alt Text](https://i.postimg.cc/Tw6WwLKp/ezgif-com-video-to-gif.gif)")

    st.markdown('''
          # About the Project \n 
            Hey this is Sani and Chaity from Gait analyzer!. \n
           
            As there was no publicly available dataset, this model was trained on a **Custom Dataset** which will soon be found in Kaggle!\n

             
            ''')
elif app_mode == 'LessDO IT':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.button('Use Webcam')
    record = st.checkbox("Record video")
    if record:
        st.checkbox("Recording", value=True)

    st.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # max faces
    detection_confidence = st.slider(
        'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.slider(
        'Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        #     if use_webcam:
        vid = cv2.VideoCapture(1)
    #     else:
    #         vid = cv2.VideoCapture(DEMO_VIDEO)
    #         tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.videoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.text('Input video')
    st.video(tfflie.name)
    fps = 0
    i = 0

    colors = [(16, 117, 245), (200, 103, 27), (16, 117, 245),
              (200, 103, 27), (16, 117, 245), (200, 103, 27)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60+num*40),
                          (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return output_frame

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    kpi1, kpi3 = st.columns(2)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_pose.Pose(
            smooth_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
    ) as pose:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            # Make detections
            image, results = multiple_detection(frame, pose)
            print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-130:]

            if len(sequence) == 130:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                #st.checkbox("Recording", value=True)
                out.write(image)
            # Dashboard
            kpi1_text.write(
                f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(
                f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            image = image_resize(image=image, width=640)
            stframe.image(image, channels='BGR', use_column_width=True)

    st.text('video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()
