import numpy as np
from flask import Flask, render_template, request, jsonify
import mediapipe as mp
import joblib
from collections import deque
import traceback

app = Flask(__name__)

# Load your trained model
model = joblib.load('static/model/yoga_pose_model.pkl')

# Pose mapping to display names
POSE_MAPPING = {
    "ArdhaChandrasana": "Half Moon Pose",
    "BaddhaKonasana": "Butterfly Pose",
    "Downward_dog": "Downward Dog",
    "Natarajasana": "Dancer Pose",
    "Triangle": "Triangle Pose",
    "UtkataKonasana": "Goddess Pose",
    "Veerabhadrasana": "Warrior Pose",
    "Vrukshasana": "Tree Pose"
}

# Pose-specific feedback
POSE_FEEDBACK = {
    "Half Moon Pose": ["Keep your standing leg straight", "Extend through your lifted heel"],
    "Butterfly Pose": ["Gently bounce your knees toward the floor", "Keep your spine straight"],
    "Downward Dog": ["Press firmly through your palms", "Rotate your upper arms outward"],
    "Dancer Pose": ["Focus on a fixed point", "Lift your chest and extend your leg"],
    "Triangle Pose": ["Keep both sides of your waist equally long", "Extend through your top arm"],
    "Goddess Pose": ["Sink your hips low", "Keep knees aligned over ankles"],
    "Warrior Pose": ["Keep your front knee at 90°", "Keep your back leg strong and straight"],
    "Tree Pose": ["Press your foot firmly into your inner thigh", "Bring palms together at heart center"]
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Smoothing predictions
prediction_history = deque(maxlen=10)  # Last 10 predictions

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def extract_angles(landmarks):
    """Extract all 14 angles used during training"""
    angles = []
    
    # Get MediaPipe index values
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    NOSE = mp_pose.PoseLandmark.NOSE.value

    # 1. Left elbow angle
    angles.append(calculate_angle(
        [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']],
        [landmarks[LEFT_ELBOW]['x'], landmarks[LEFT_ELBOW]['y']],
        [landmarks[LEFT_WRIST]['x'], landmarks[LEFT_WRIST]['y']]
    ))
    
    # 2. Right elbow angle
    angles.append(calculate_angle(
        [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']],
        [landmarks[RIGHT_ELBOW]['x'], landmarks[RIGHT_ELBOW]['y']],
        [landmarks[RIGHT_WRIST]['x'], landmarks[RIGHT_WRIST]['y']]
    ))
    
    # 3. Left shoulder angle
    angles.append(calculate_angle(
        [landmarks[LEFT_ELBOW]['x'], landmarks[LEFT_ELBOW]['y']],
        [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']],
        [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']]
    ))
    
    # 4. Right shoulder angle
    angles.append(calculate_angle(
        [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']],
        [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']],
        [landmarks[RIGHT_ELBOW]['x'], landmarks[RIGHT_ELBOW]['y']]
    ))
    
    # 5. Left knee angle
    angles.append(calculate_angle(
        [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']],
        [landmarks[LEFT_KNEE]['x'], landmarks[LEFT_KNEE]['y']],
        [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']]
    ))
    
    # 6. Right knee angle
    angles.append(calculate_angle(
        [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']],
        [landmarks[RIGHT_KNEE]['x'], landmarks[RIGHT_KNEE]['y']],
        [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']]
    ))
    
    # 7. Angle for Ardha Chandrasana 1
    angles.append(calculate_angle(
        [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']],
        [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']],
        [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']]
    ))
    
    # 8. Angle for Ardha Chandrasana 2
    angles.append(calculate_angle(
        [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']],
        [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']],
        [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']]
    ))
    
    # 9. Hand angle
    angles.append(calculate_angle(
        [landmarks[LEFT_ELBOW]['x'], landmarks[LEFT_ELBOW]['y']],
        [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']],
        [landmarks[RIGHT_ELBOW]['x'], landmarks[RIGHT_ELBOW]['y']]
    ))
    
    # 10. Left hip angle
    angles.append(calculate_angle(
        [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']],
        [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']],
        [landmarks[LEFT_KNEE]['x'], landmarks[LEFT_KNEE]['y']]
    ))
    
    # 11. Right hip angle
    angles.append(calculate_angle(
        [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']],
        [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']],
        [landmarks[RIGHT_KNEE]['x'], landmarks[RIGHT_KNEE]['y']]
    ))
    
    # 12. Neck angle
    angles.append(calculate_angle(
        [landmarks[NOSE]['x'], landmarks[NOSE]['y']],
        [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']],
        [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']]
    ))
    
    # 13. Left wrist angle
    angles.append(calculate_angle(
        [landmarks[LEFT_WRIST]['x'], landmarks[LEFT_WRIST]['y']],
        [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']],
        [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']]
    ))
    
    # 14. Right wrist angle
    angles.append(calculate_angle(
        [landmarks[RIGHT_WRIST]['x'], landmarks[RIGHT_WRIST]['y']],
        [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']],
        [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']]
    ))
    
    return angles

@app.route('/')
def index():
    """Landing page with corrected pose names"""
    return render_template('index.html', poses=POSE_MAPPING)

@app.route('/coach')
def coach():
    """Real-time coaching interface"""
    return render_template('coach.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for pose prediction"""
    try:
        data = request.get_json()
        landmarks = data['landmarks']
        
        # Validate landmarks
        if not landmarks or len(landmarks) < 33:
            return jsonify({
                'pose': 'Waiting...', 
                'feedback': ['Please position yourself in frame'], 
                'confidence': 0
            })
        
        # Convert landmarks to proper format
        mp_landmarks = []
        for lm in landmarks:
            mp_landmarks.append({
                'x': lm[0],
                'y': lm[1],
                'z': lm[2] if len(lm) > 2 else 0
            })
        
        # Extract angles
        angles = extract_angles(mp_landmarks)
        
        # Predict pose
        pose_label = model.predict([angles])[0]
        pose_name = POSE_MAPPING.get(pose_label, pose_label)
        
        # Calculate confidence
        proba = model.predict_proba([angles])[0]
        confidence = np.max(proba)
        
        # Add to history for smoothing
        prediction_history.append(pose_label)
        
        # Get most frequent recent prediction
        if prediction_history:
            final_prediction = max(set(prediction_history), key=prediction_history.count)
        else:
            final_prediction = pose_label
        
        # Generate personalized feedback
        feedback = POSE_FEEDBACK.get(pose_name, [])
        
        return jsonify({
            'pose': pose_name,
            'feedback': feedback,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'pose': 'Error', 
            'feedback': ['System processing error'], 
            'confidence': 0
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)