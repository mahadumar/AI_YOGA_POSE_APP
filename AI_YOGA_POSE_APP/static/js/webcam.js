const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

// Global reference to latest results
let latestResults = null;

// Initialize MediaPipe Pose
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});

pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

pose.onResults(onPoseResults);

// Start webcam
const camera = new Camera(videoElement, {
    onFrame: async () => {
        try {
            await pose.send({image: videoElement});
        } catch (e) {
            console.log("Camera error:", e);
        }
    },
    width: 1280,
    height: 720
});
camera.start();

function onPoseResults(results) {
    latestResults = results; // Store globally
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw camera feed
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.poseLandmarks) {
        // Draw pose landmarks
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, 
                      {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, results.poseLandmarks, 
                     {color: '#FF0000', lineWidth: 2, radius: 4});
        
        // Prepare data for prediction
        const landmarks = [];
        for (const landmark of results.poseLandmarks) {
            landmarks.push([landmark.x, landmark.y, landmark.z]);
        }
        
        // Send to Flask backend
        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({landmarks})
        })
        .then(response => {
            if (!response.ok) throw new Error('Network error');
            return response.json();
        })
        .then(handlePrediction)
        .catch(error => {
            console.error('Prediction failed:', error);
            document.getElementById('pose-result').textContent = 'Error';
        });
    }
    canvasCtx.restore();
}

function handlePrediction(data) {
    document.getElementById('pose-result').textContent = data.pose;
    
    // Update confidence meter
    document.querySelector('.confidence-fill').style.width = `${data.confidence * 100}%`;
    
    // Generate feedback if we have pose data
    if (latestResults && latestResults.poseLandmarks && data.pose !== 'Waiting...') {
        window.analyzePose(latestResults.poseLandmarks, data.pose);
    }
}