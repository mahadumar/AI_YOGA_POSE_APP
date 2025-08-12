// feedback.js - Real-time pose comparison and feedback
const POSE_THRESHOLDS = {
    "Half Moon Pose": {
        angles: [170, 160, 175, 170, 165, 175, 170, 165],
        tolerance: 15
    },
    "Butterfly Pose": {
        angles: [120, 125, 110, 115, 100, 105, 90, 95],
        tolerance: 20
    },
    "Downward Dog": {
        angles: [145, 150, 155, 160, 170, 165, 175, 170],
        tolerance: 15
    },
    "Dancer Pose": {
        angles: [165, 170, 160, 155, 150, 145, 140, 135],
        tolerance: 15
    },
    "Triangle Pose": {
        angles: [155, 160, 150, 145, 140, 135, 130, 125],
        tolerance: 15
    },
    "Goddess Pose": {
        angles: [120, 125, 130, 135, 140, 145, 150, 155],
        tolerance: 20
    },
    "Warrior Pose": {
        angles: [145, 150, 155, 160, 165, 170, 175, 170],
        tolerance: 15
    },
    "Tree Pose": {
        angles: [170, 165, 160, 155, 150, 145, 140, 135],
        tolerance: 10
    }
};

const JOINT_NAMES = [
    "Left elbow", 
    "Right elbow",
    "Left shoulder",
    "Right shoulder",
    "Left hip",
    "Right hip",
    "Left knee",
    "Right knee"
];

// Calculate angle between three points
function calculateAngle(a, b, c) {
    const ab = [b.x - a.x, b.y - a.y];
    const bc = [c.x - b.x, c.y - b.y];
    
    const dot = ab[0] * bc[0] + ab[1] * bc[1];
    const magAB = Math.sqrt(ab[0]**2 + ab[1]**2);
    const magBC = Math.sqrt(bc[0]**2 + bc[1]**2);
    
    let angle = Math.acos(dot / (magAB * magBC));
    angle = angle * (180 / Math.PI);
    
    return Math.round(angle);
}

// Process landmarks and extract key angles
function processLandmarks(landmarks) {
    return [
        // Left elbow (11,13,15)
        calculateAngle(landmarks[11], landmarks[13], landmarks[15]),
        // Right elbow (12,14,16)
        calculateAngle(landmarks[12], landmarks[14], landmarks[16]),
        // Left shoulder (13,11,23)
        calculateAngle(landmarks[13], landmarks[11], landmarks[23]),
        // Right shoulder (14,12,24)
        calculateAngle(landmarks[14], landmarks[12], landmarks[24]),
        // Left hip (11,23,25)
        calculateAngle(landmarks[11], landmarks[23], landmarks[25]),
        // Right hip (12,24,26)
        calculateAngle(landmarks[12], landmarks[24], landmarks[26]),
        // Left knee (23,25,27)
        calculateAngle(landmarks[23], landmarks[25], landmarks[27]),
        // Right knee (24,26,28)
        calculateAngle(landmarks[24], landmarks[26], landmarks[28])
    ];
}

// Generate personalized feedback based on deviations
function generateFeedback(currentPose, userAngles) {
    const ideal = POSE_THRESHOLDS[currentPose];
    if (!ideal) return [];
    
    const feedback = [];
    
    userAngles.forEach((angle, index) => {
        const deviation = Math.abs(angle - ideal.angles[index]);
        
        if (deviation > ideal.tolerance) {
            const direction = angle > ideal.angles[index] ? "decrease" : "increase";
            feedback.push({
                joint: JOINT_NAMES[index],
                current: angle,
                ideal: ideal.angles[index],
                instruction: `${direction} angle by ${deviation}°`
            });
        }
    });
    
    return feedback;
}

// Main function to handle pose analysis
function analyzePose(landmarks, currentPose) {
    try {
        const userAngles = processLandmarks(landmarks);
        const feedback = generateFeedback(currentPose, userAngles);
        
        // Update UI with feedback
        const feedbackList = document.getElementById('feedback-list');
        feedbackList.innerHTML = '';
        
        if (feedback.length === 0) {
            const li = document.createElement('li');
            li.textContent = "Perfect form! Maintain this pose";
            feedbackList.appendChild(li);
        } else {
            feedback.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <strong>${item.joint}:</strong>
                    <span class="deviation">${item.current}° vs ${item.ideal}°</span>
                    <span class="instruction">→ ${item.instruction}</span>
                `;
                feedbackList.appendChild(li);
            });
        }
        
        updatePoseGuidance(currentPose);
    } catch (e) {
        console.error("Feedback error:", e);
    }
}

// Show pose-specific guidance
function updatePoseGuidance(pose) {
    const instructions = document.getElementById('pose-instructions');
    
    const GUIDANCE = {
        "Half Moon Pose": "Keep your standing leg straight and extend through your lifted heel",
        "Butterfly Pose": "Gently bounce your knees toward the floor while keeping your spine straight",
        "Downward Dog": "Press firmly through your palms and rotate your upper arms outward",
        "Dancer Pose": "Focus on a fixed point to maintain balance while lifting your chest",
        "Triangle Pose": "Keep both sides of your waist equally long and extend through your top arm",
        "Goddess Pose": "Sink your hips low while keeping your knees aligned over your ankles",
        "Warrior Pose": "Keep your front knee at 90° and your back leg strong and straight",
        "Tree Pose": "Press your foot firmly into your inner thigh and bring palms together at heart center"
    };
    
    instructions.textContent = GUIDANCE[pose] || 
        "Focus on steady breathing and maintain the pose alignment";
}

// Export functions for use in webcam.js
window.analyzePose = analyzePose;