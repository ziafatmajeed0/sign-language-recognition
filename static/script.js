const video = document.getElementById('video');
const startButton = document.getElementById('start-camera');
const stopButton = document.getElementById('stop-camera');
const feedback = document.getElementById('feedback');
const cameraContainer = document.getElementById('camera-container');
const spinner = document.getElementById('loading-spinner');

let stream = null;

// Start Camera
startButton.addEventListener('click', async () => {
    feedback.innerText = "Initializing camera...";
    spinner.style.display = "block";
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;  // Set the video stream source
        startButton.disabled = true;
        stopButton.disabled = false;
        feedback.innerText = "Camera started!";
        cameraContainer.classList.add('active');
    } catch (err) {
        console.error("Error starting camera:", err);  // Log the error if any
        feedback.innerText = "Error starting camera.";
    } finally {
        spinner.style.display = "none";
    }
});

// Stop Camera
stopButton.addEventListener('click', () => {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
    }
    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    feedback.innerText = "Camera stopped.";
    cameraContainer.classList.remove('active');
});
